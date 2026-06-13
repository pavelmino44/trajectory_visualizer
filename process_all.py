import os
import json
import numpy as np
import pandas as pd
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from branca.colormap import LinearColormap
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


# ==========  Наш класс навигации  ==========
class KalmanFilterGPS:
    def __init__(self, dt: float):
        self.dt = dt
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] = np.eye(3) * 0.1
        self.Q[3:6, 3:6] = np.eye(3) * 0.5
        self.R = np.eye(3) * 4.0
        self.x = np.zeros(6)
        self.P = np.eye(6) * 100

    def predict(self, accel: np.ndarray):
        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) * self.dt
        self.x = self.F @ self.x + B @ accel
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, gps_pos: np.ndarray, gps_accuracy: float = 5.0):
        if gps_accuracy > 0:
            self.R = np.eye(3) * (gps_accuracy ** 2)
        y = gps_pos - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_state(self):
        return {
            'position': self.x[0:3].copy(),
            'velocity': self.x[3:6].copy()
        }


class IMUNavigation:
    def __init__(self, dt: float, init_position: np.ndarray = None,
                 init_orientation: np.ndarray = None):
        self.dt = dt
        self.pos = init_position.copy() if init_position is not None else np.zeros(3)
        self.vel = np.zeros(3)

        if init_orientation is not None:
            r = R.from_euler('zyx', init_orientation)
            quat_scipy = r.as_quat()
            self.quat = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])
        else:
            self.quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.kf = KalmanFilterGPS(dt)
        if init_position is not None:
            self.kf.x[0:3] = init_position.copy()

        # Для unwrap углов
        self.last_yaw_rad = 0.0
        self.last_pitch_rad = 0.0
        self.last_roll_rad = 0.0
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.roll_offset = 0.0

    def quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    def quat_conjugate(self, q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]])

    def quat_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        qv = np.array([0.0, v[0], v[1], v[2]])
        q_conj = self.quat_conjugate(q)
        result = self.quat_multiply(self.quat_multiply(q, qv), q_conj)
        return np.array([result[1], result[2], result[3]])

    def update_orientation(self, gyro: np.ndarray):
        omega = np.array([0.0, gyro[0], gyro[1], gyro[2]])
        dq_dt = 0.5 * self.quat_multiply(self.quat, omega)
        self.quat = self.quat + dq_dt * self.dt
        self.quat = self.quat / np.linalg.norm(self.quat)

    def get_euler_angles(self) -> np.ndarray:
        r = R.from_quat([self.quat[1], self.quat[2], self.quat[3], self.quat[0]])
        return r.as_euler('zyx')

    def _unwrap_angle(self, current_rad: float, last_rad: float, offset: float):
        diff = current_rad - last_rad
        if diff > np.pi:
            offset -= 2 * np.pi
        elif diff < -np.pi:
            offset += 2 * np.pi
        unwrapped = current_rad + offset
        return unwrapped, current_rad, offset

    def update(self, accel: np.ndarray, gyro: np.ndarray, gps_data: dict = None) -> dict:
        self.update_orientation(gyro)
        accel_global = self.quat_rotate(self.quat, accel)
        gravity = np.array([0.0, 0.0, 9.81])
        linear_accel = accel_global - gravity

        self.vel = self.vel + linear_accel * self.dt
        self.pos = self.pos + self.vel * self.dt
        self.kf.predict(linear_accel)

        if gps_data is not None:
            self.kf.update(gps_data['pos'], gps_data.get('accuracy', 5.0))

        state = self.kf.get_state()
        euler = self.get_euler_angles()

        yaw_unwrapped, self.last_yaw_rad, self.yaw_offset = self._unwrap_angle(
            euler[0], self.last_yaw_rad, self.yaw_offset
        )
        pitch_unwrapped, self.last_pitch_rad, self.pitch_offset = self._unwrap_angle(
            euler[1], self.last_pitch_rad, self.pitch_offset
        )
        roll_unwrapped, self.last_roll_rad, self.roll_offset = self._unwrap_angle(
            euler[2], self.last_roll_rad, self.roll_offset
        )

        return {
            'pos_fused': state['position'],
            'vel_fused': state['velocity'],
            'yaw_rad': yaw_unwrapped,
            'pitch_rad': pitch_unwrapped,
            'roll_rad': roll_unwrapped,
            'yaw_deg': np.degrees(yaw_unwrapped),
            'pitch_deg': np.degrees(pitch_unwrapped),
            'roll_deg': np.degrees(roll_unwrapped)
        }


# ==========  Вспомогательные функции ==========
def pressure_to_altitude(pressure_hpa, ref_pressure=1013.25):
    return 44330.0 * (1.0 - (pressure_hpa / ref_pressure) ** 0.1903)


def load_experiment(folder):
    imu = pd.read_csv(os.path.join(folder, 'imu.csv'))
    imu.columns = [col.split('[')[0] for col in imu.columns]

    pressure = pd.read_csv(os.path.join(folder, 'pressure.csv'))
    pressure.columns = [col.split('[')[0] for col in pressure.columns]

    gnss = pd.read_csv(os.path.join(folder, 'Location.csv'))
    gnss.columns = [col.split('[')[0] for col in gnss.columns]

    return imu, pressure, gnss


def llh_to_enu(lat, lon, h, lat0, lon0, h0):
    dlat = (lat - lat0) * 111111.0
    dlon = (lon - lon0) * 111111.0 * np.cos(np.deg2rad(lat0))
    return np.array([dlon, dlat, h - h0])


# ==========  Функции сохранения графиков ==========

def save_trajectory_plot(folder, results_df):
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(results_df['pos_x'], results_df['pos_y'], 'b-', linewidth=2)
    ax.scatter(results_df['pos_x'].iloc[0], results_df['pos_y'].iloc[0],
               c='green', marker='o', s=100, label='Старт')
    ax.scatter(results_df['pos_x'].iloc[-1], results_df['pos_y'].iloc[-1],
               c='red', marker='s', s=100, label='Финиш')
    ax.set_xlabel('X (Восток) [м]')
    ax.set_ylabel('Y (Север) [м]')
    ax.set_title('Траектория XY')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    out_path = os.path.join(folder, 'trajectory_xy.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Траектория XY сохранена: {out_path}")


def save_position_plots(folder, results_df):
    time = results_df['time'] - results_df['time'].iloc[0]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(time, results_df['pos_x'], 'b-', linewidth=1)
    axes[0].set_ylabel('X [м]')
    axes[0].set_title('Позиция X (Восток)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, results_df['pos_y'], 'b-', linewidth=1)
    axes[1].set_ylabel('Y [м]')
    axes[1].set_title('Позиция Y (Север)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, results_df['pos_z'], 'b-', linewidth=1)
    axes[2].set_xlabel('Время [с]')
    axes[2].set_ylabel('Z [м]')
    axes[2].set_title('Высота Z (Вверх)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(folder, 'positions_time.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Графики позиций сохранены: {out_path}")


def save_angles_plot(folder, results_df):
    time = results_df['time'] - results_df['time'].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(time, results_df['yaw_deg'], 'c-', linewidth=1.5, label='Рысканье (Yaw)')
    ax.plot(time, results_df['pitch_deg'], 'orange', linewidth=1.5, label='Тангаж (Pitch)')
    ax.plot(time, results_df['roll_deg'], 'purple', linewidth=1.5, label='Крен (Roll)')
    ax.set_xlabel('Время [с]')
    ax.set_ylabel('Угол [градусы]')
    ax.set_title('Ориентация (непрерывная)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(folder, 'angles_continuous.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ График углов сохранён: {out_path}")


def save_imu_plots(folder, imu, t0):
    """Сохраняет графики IMU с отдельными полотнами для каждой оси гироскопа"""
    t_rel = imu['t_utc'].values - t0

    # Графики акселерометра (3 оси на одном полотне)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_rel, imu[['a_x', 'a_y', 'a_z']].values)
    ax.set_ylabel('Ускорение (м/с²)')
    ax.set_xlabel('Время (с)')
    ax.legend(['a_x', 'a_y', 'a_z'])
    ax.set_title('Акселерометр')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(folder, 'accelerometer.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Акселерометр сохранён: {out_path}")

    # Графики гироскопа - отдельные полотна для каждой оси
    # Вычисляем общий диапазон для всех осей гироскопа
    gyro_data = imu[['w_x', 'w_y', 'w_z']].values
    gyro_min = np.min(gyro_data)
    gyro_max = np.max(gyro_data)
    gyro_range = max(abs(gyro_min), abs(gyro_max))
    y_lim = (-gyro_range * 1.1, gyro_range * 1.1)  # 10% запас

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Ось X (крен)
    axes[0].plot(t_rel, imu['w_x'], 'r-', linewidth=1)
    axes[0].set_ylabel('ω_x (рад/с)')
    axes[0].set_title('Гироскоп - Ось X (Крен/Roll)')
    axes[0].set_ylim(y_lim)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    # Ось Y (тангаж)
    axes[1].plot(t_rel, imu['w_y'], 'g-', linewidth=1)
    axes[1].set_ylabel('ω_y (рад/с)')
    axes[1].set_title('Гироскоп - Ось Y (Тангаж/Pitch)')
    axes[1].set_ylim(y_lim)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    # Ось Z (рыскание)
    axes[2].plot(t_rel, imu['w_z'], 'b-', linewidth=1)
    axes[2].set_ylabel('ω_z (рад/с)')
    axes[2].set_xlabel('Время (с)')
    axes[2].set_title('Гироскоп - Ось Z (Рыскание/Yaw)')
    axes[2].set_ylim(y_lim)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(folder, 'gyroscope_axes.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Гироскоп (3 оси, единый масштаб) сохранён: {out_path}")


def save_combined_plot(folder, results_df):
    time = results_df['time'] - results_df['time'].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(results_df['pos_x'], results_df['pos_y'], 'b-', linewidth=2)
    axes[0, 0].scatter(results_df['pos_x'].iloc[0], results_df['pos_y'].iloc[0],
                       c='green', marker='o', s=100, label='Старт')
    axes[0, 0].scatter(results_df['pos_x'].iloc[-1], results_df['pos_y'].iloc[-1],
                       c='red', marker='s', s=100, label='Финиш')
    axes[0, 0].set_xlabel('X (Восток) [м]')
    axes[0, 0].set_ylabel('Y (Север) [м]')
    axes[0, 0].set_title('Траектория XY')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')

    axes[0, 1].plot(time, results_df['pos_x'], 'b-', linewidth=1)
    axes[0, 1].set_xlabel('Время [с]')
    axes[0, 1].set_ylabel('X [м]')
    axes[0, 1].set_title('Позиция X')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time, results_df['pos_y'], 'b-', linewidth=1)
    axes[1, 0].set_xlabel('Время [с]')
    axes[1, 0].set_ylabel('Y [м]')
    axes[1, 0].set_title('Позиция Y')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(time, results_df['yaw_deg'], 'c-', linewidth=1.5, label='Yaw')
    axes[1, 1].plot(time, results_df['pitch_deg'], 'orange', linewidth=1.5, label='Pitch')
    axes[1, 1].plot(time, results_df['roll_deg'], 'purple', linewidth=1.5, label='Roll')
    axes[1, 1].set_xlabel('Время [с]')
    axes[1, 1].set_ylabel('Угол [градусы]')
    axes[1, 1].set_title('Ориентация')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('IMU+GPS Навигация', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(folder, 'combined_navigation.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Комбинированный график сохранён: {out_path}")


# ==========  Генератор HTML плеера ==========

def generate_html_player(folder, video_offset, lats, lons, t_utc,
                         yaw_deg, pitch_deg, roll_deg):
    step = max(1, len(t_utc) // 2000)

    points_js = json.dumps([{
        "t": float(t_utc[i]),
        "lat": float(lats[i]),
        "lon": float(lons[i]),
        "yaw": float(yaw_deg[i]),
        "pitch": float(pitch_deg[i]),
        "roll": float(roll_deg[i])
    } for i in range(0, len(t_utc), step)])

    video_start = t_utc[0] + video_offset

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>Навигация с видео</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            height: 100vh;
            overflow: hidden;
        }}

        .main-container {{
            display: flex;
            height: 100vh;
        }}

        .left-column {{
            width: 60%;
            display: flex;
            flex-direction: column;
        }}

        #map {{
            height: calc(100% - 100px);
            min-height: 300px;
        }}

        .angles-panel {{
            height: 100px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            display: flex;
            justify-content: space-around;
            align-items: center;
            padding: 10px 20px;
            gap: 20px;
            border-top: 2px solid #00adb5;
        }}

        .angle-card {{
            flex: 1;
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 8px 15px;
            text-align: center;
            backdrop-filter: blur(10px);
            transition: transform 0.2s;
        }}

        .angle-card:hover {{
            transform: translateY(-2px);
            background: rgba(255,255,255,0.15);
        }}

        .angle-label {{
            font-size: 12px;
            color: #aaa;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }}

        .angle-value {{
            font-size: 28px;
            font-weight: bold;
            font-family: monospace;
        }}

        .yaw-value {{ color: #4dc9f6; }}
        .pitch-value {{ color: #f67019; }}
        .roll-value {{ color: #f95a5a; }}

        .angle-unit {{
            font-size: 12px;
            color: #666;
        }}

        .right-column {{
            width: 40%;
            background: #0f0f1a;
            display: flex;
            flex-direction: column;
            padding: 20px;
            gap: 15px;
        }}

        .video-container {{
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        video {{
            width: 100%;
            display: block;
        }}

        .slider-container {{
            width: 100%;
        }}

        input[type="range"] {{
            width: 100%;
            cursor: pointer;
            background: #2c2c3e;
            height: 4px;
            border-radius: 2px;
            -webkit-appearance: none;
        }}

        input[type="range"]:focus {{
            outline: none;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #00adb5;
            cursor: pointer;
        }}

        .time-info {{
            font-family: monospace;
            font-size: 12px;
            color: #888;
            text-align: center;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 8px;
        }}

        .time-info span {{
            color: #00adb5;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="left-column">
            <div id="map"></div>
            <div class="angles-panel">
                <div class="angle-card">
                    <div class="angle-label">РЫСКАНИЕ</div>
                    <div class="angle-value yaw-value" id="yaw-val">0.0<span class="angle-unit">°</span></div>
                </div>
                <div class="angle-card">
                    <div class="angle-label">ТАНГАЖ</div>
                    <div class="angle-value pitch-value" id="pitch-val">0.0<span class="angle-unit">°</span></div>
                </div>
                <div class="angle-card">
                    <div class="angle-label">КРЕН</div>
                    <div class="angle-value roll-value" id="roll-val">0.0<span class="angle-unit">°</span></div>
                </div>
            </div>
        </div>
        <div class="right-column">
            <div class="video-container">
                <video id="video" controls>
                    <source src="video.mp4" type="video/mp4">
                </video>
            </div>
            <div class="slider-container">
                <input type="range" id="slider" min="0" max="100" value="0" step="0.01">
            </div>
            <div class="time-info">
                <span id="video-time">0.00</span> с / <span id="total-time">0</span> с
            </div>
        </div>
    </div>

    <script>
        const trajectory = {points_js};
        const videoStartUtc = {video_start};
        const totalPoints = trajectory.length;

        let map, marker, polyline;
        let video = document.getElementById('video');
        let slider = document.getElementById('slider');

        function initMap() {{
            if (!trajectory || trajectory.length === 0) return;

            const lats = trajectory.map(p => p.lat);
            const lons = trajectory.map(p => p.lon);
            const bounds = [[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]];

            map = L.map('map').fitBounds(bounds);

            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap',
                maxZoom: 19
            }}).addTo(map);

            const latlngs = trajectory.map(p => [p.lat, p.lon]);
            polyline = L.polyline(latlngs, {{color: '#00adb5', weight: 3, opacity: 0.8}}).addTo(map);

            marker = L.marker([trajectory[0].lat, trajectory[0].lon], {{
                icon: L.divIcon({{html: '📍', iconSize: [24, 24], className: 'custom-marker'}})
            }}).addTo(map);
        }}

        function findNearestPoint(utcTime) {{
            let left = 0, right = trajectory.length - 1;
            while (left < right) {{
                const mid = Math.floor((left + right) / 2);
                if (trajectory[mid].t < utcTime) left = mid + 1;
                else right = mid;
            }}
            const idx = Math.max(0, Math.min(left, trajectory.length - 1));
            if (idx > 0 && Math.abs(trajectory[idx-1].t - utcTime) < Math.abs(trajectory[idx].t - utcTime)) {{
                return {{ point: trajectory[idx-1], index: idx-1 }};
            }}
            return {{ point: trajectory[idx], index: idx }};
        }}

        function updateByTime(currentVideoTime) {{
            const utcTime = videoStartUtc + currentVideoTime;
            const {{ point, index }} = findNearestPoint(utcTime);

            if (marker) {{
                marker.setLatLng([point.lat, point.lon]);
                map.panTo([point.lat, point.lon], {{ animate: true, duration: 0.2 }});
            }}

            document.getElementById('yaw-val').innerHTML = point.yaw.toFixed(1) + '<span class="angle-unit">°</span>';
            document.getElementById('pitch-val').innerHTML = point.pitch.toFixed(1) + '<span class="angle-unit">°</span>';
            document.getElementById('roll-val').innerHTML = point.roll.toFixed(1) + '<span class="angle-unit">°</span>';

            document.getElementById('video-time').textContent = currentVideoTime.toFixed(2);
        }}

        function onTimeUpdate() {{
            const currentVideoTime = video.currentTime;
            slider.value = currentVideoTime;
            updateByTime(currentVideoTime);
        }}

        function onSliderChange() {{
            const newTime = parseFloat(slider.value);
            if (!isNaN(newTime)) {{
                video.currentTime = newTime;
                updateByTime(newTime);
            }}
        }}

        function init() {{
            initMap();

            video.addEventListener('loadedmetadata', () => {{
                slider.max = video.duration;
                document.getElementById('total-time').textContent = video.duration.toFixed(2);
            }});

            video.addEventListener('timeupdate', onTimeUpdate);
            slider.addEventListener('input', onSliderChange);

            setTimeout(() => {{
                if (video.duration > 0) {{
                    updateByTime(0);
                }}
            }}, 100);
        }}

        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', init);
        }} else {{
            init();
        }}
    </script>
</body>
</html>"""

    out_path = os.path.join(folder, 'player.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"✅ HTML-плеер сохранён: {out_path}")


def generate_folium_map(folder, lats, lons, alts):
    norm = colors.Normalize(vmin=alts.min(), vmax=alts.max())
    colormap = cm.ScalarMappable(norm=norm, cmap='coolwarm')
    colors_hex = [colors.to_hex(colormap.to_rgba(v)) for v in alts]

    center = [lats.mean(), lons.mean()]
    m = folium.Map(location=center, zoom_start=17)

    for i in range(len(lats) - 1):
        folium.PolyLine(
            locations=[(lats[i], lons[i]), (lats[i + 1], lons[i + 1])],
            color=colors_hex[i],
            weight=5,
            opacity=0.8
        ).add_to(m)

    folium.Marker([lats[0], lons[0]], popup='Старт',
                  icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([lats[-1], lons[-1]], popup='Финиш',
                  icon=folium.Icon(color='red')).add_to(m)

    colormap_branca = LinearColormap(
        colors=['blue', 'white', 'red'],
        vmin=alts.min(), vmax=alts.max(),
        caption='Высота (м)'
    )
    m.add_child(colormap_branca)

    out_path = os.path.join(folder, 'map_folium.html')
    m.save(out_path)
    print(f"✅ Folium-карта сохранена: {out_path}")


def plot_3d_trajectory(folder, positions, t_utc, t0):
    t_rel = t_utc - t0
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                    c=t_rel, cmap='viridis', s=2, alpha=0.8)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color='grey', linewidth=0.5, alpha=0.4)

    ax.scatter(*positions[0], color='green', s=60, label='Старт', edgecolors='black')
    ax.scatter(*positions[-1], color='red', s=60, label='Финиш', edgecolors='black')

    ax.set_xlabel('Восток (м)')
    ax.set_ylabel('Север (м)')
    ax.set_zlabel('Высота (м)')
    ax.set_title('Трёхмерная траектория (EKF)')
    ax.legend()

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Время (с)')

    plt.tight_layout()
    out_path = os.path.join(folder, 'trajectory_3d.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ 3D траектория сохранена: {out_path}")


# ==========  Обработка одного эксперимента  ==========
def process_single(folder, video_offset=0.0):
    print(f"\n{'=' * 60}")
    print(f"Обрабатываю {folder}...")
    print('=' * 60)

    imu, pressure, gnss = load_experiment(folder)
    t0 = imu['t_utc'].iloc[0]
    dt = np.median(np.diff(imu['t_utc'].values))
    print(f"   Частота IMU: {1 / dt:.1f} Гц")

    gnss_valid = gnss.dropna(subset=['latitude', 'longitude', 'altitudeAboveMeanSeaLevel'])
    lat0 = gnss_valid['latitude'].iloc[0]
    lon0 = gnss_valid['longitude'].iloc[0]
    h0 = gnss_valid['altitudeAboveMeanSeaLevel'].iloc[0]

    print(f"   Референсная точка: {lat0:.6f}, {lon0:.6f}")

    with open(os.path.join(folder, 'ref_point.json'), 'w') as f:
        json.dump({'lat0': lat0, 'lon0': lon0}, f)

    gnss_t = gnss_valid['time'] / 1e9
    gnss_pos = np.array([llh_to_enu(gnss_valid['latitude'].iloc[i],
                                    gnss_valid['longitude'].iloc[i],
                                    gnss_valid['altitudeAboveMeanSeaLevel'].iloc[i],
                                    lat0, lon0, h0)
                         for i in range(len(gnss_valid))])

    f_gnss_x = interp1d(gnss_t, gnss_pos[:, 0], kind='linear',
                        bounds_error=False, fill_value='extrapolate')
    f_gnss_y = interp1d(gnss_t, gnss_pos[:, 1], kind='linear',
                        bounds_error=False, fill_value='extrapolate')

    p_val = pressure['pressure'].values
    alt_baro = pressure_to_altitude(p_val, ref_pressure=p_val[0])
    alt_offset = h0 - alt_baro[0]
    alt_baro_corrected = alt_baro + alt_offset
    f_baro = interp1d(pressure['t_utc'].values, alt_baro_corrected,
                      kind='linear', bounds_error=False, fill_value='extrapolate')

    nav = IMUNavigation(dt=dt, init_position=np.zeros(3), init_orientation=np.zeros(3))

    print("\n   Обработка данных IMU+GPS...")
    results_list = []
    gps_corrections = 0

    for i in range(len(imu)):
        t_now = imu['t_utc'].iloc[i]
        accel = imu[['a_x', 'a_y', 'a_z']].iloc[i].values
        gyro = imu[['w_x', 'w_y', 'w_z']].iloc[i].values

        gps_data = None
        if i % 100 == 0:
            try:
                gps_pos = np.array([f_gnss_x(t_now), f_gnss_y(t_now), f_baro(t_now)])
                gps_data = {'pos': gps_pos, 'accuracy': 5.0}
                gps_corrections += 1
            except:
                pass

        state = nav.update(accel, gyro, gps_data)

        results_list.append({
            'time': t_now,
            'pos_x': state['pos_fused'][0],
            'pos_y': state['pos_fused'][1],
            'pos_z': state['pos_fused'][2],
            'yaw_deg': state['yaw_deg'],
            'pitch_deg': state['pitch_deg'],
            'roll_deg': state['roll_deg']
        })

    print(f"   Выполнено GPS коррекций: {gps_corrections}")

    results_df = pd.DataFrame(results_list)

    output_csv = os.path.join(folder, 'navigation_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"✅ Результаты сохранены в CSV: {output_csv}")

    cos_lat = np.cos(np.deg2rad(lat0))
    lats = lat0 + results_df['pos_y'].values / 111111.0
    lons = lon0 + results_df['pos_x'].values / (111111.0 * cos_lat)
    alts = results_df['pos_z'].values + h0

    print("\n5. Сохранение графиков...")
    save_trajectory_plot(folder, results_df)
    save_position_plots(folder, results_df)
    save_angles_plot(folder, results_df)
    save_imu_plots(folder, imu, t0)
    save_combined_plot(folder, results_df)

    positions_fused = results_df[['pos_x', 'pos_y', 'pos_z']].values
    plot_3d_trajectory(folder, positions_fused, results_df['time'].values, t0)

    generate_folium_map(folder, lats, lons, alts)
    generate_html_player(folder, video_offset, lats, lons,
                         results_df['time'].values,
                         results_df['yaw_deg'].values,
                         results_df['pitch_deg'].values,
                         results_df['roll_deg'].values)

    print(f"\n✅ Обработка {folder} завершена!")


def process_all(base_dir='data'):
    subdirs = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('experiment_')
    ])

    if not subdirs:
        print("❌ Папки experiment_* не найдены в data/")
        return

    print(f"Найдено экспериментов: {len(subdirs)}")

    for sub in subdirs:
        folder = os.path.join(base_dir, sub)

        offset = 0.0
        info_path = os.path.join(folder, 'video_info.txt')
        if os.path.exists(info_path):
            with open(info_path) as f:
                video_start_unix = float(f.read().strip())
            imu_temp = pd.read_csv(os.path.join(folder, 'imu.csv'))
            imu_temp.columns = [c.split('[')[0] for c in imu_temp.columns]
            t0_data = imu_temp['t_utc'].iloc[0]
            offset = video_start_unix - t0_data
            print(f"   Смещение видео: {offset:.2f} с")

        process_single(folder, video_offset=offset)


if __name__ == '__main__':
    process_all()