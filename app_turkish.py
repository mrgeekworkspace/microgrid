import streamlit as st
import numpy as np
import random
import time
import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AI Destekli AC Mikro Şebeke Kontrol Sistemi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
    .control-section {
        background: #ffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'secondary_ai_enabled' not in st.session_state:
        st.session_state.secondary_ai_enabled = True
    if 'tertiary_ai_enabled' not in st.session_state:
        st.session_state.tertiary_ai_enabled = True
    if 'island_mode' not in st.session_state:
        st.session_state.island_mode = False
    if 'battery_soc' not in st.session_state:
        st.session_state.battery_soc = 80.0
    if 'voltage' not in st.session_state:
        st.session_state.voltage = 230.0
    if 'frequency' not in st.session_state:
        st.session_state.frequency = 50.0
    if 'pv_output' not in st.session_state:
        st.session_state.pv_output = 3.0
    if 'load_demand' not in st.session_state:
        st.session_state.load_demand = 3.5
    if 'grid_power' not in st.session_state:
        st.session_state.grid_power = 0.5
    if 'data_history' not in st.session_state:
        st.session_state.data_history = {
            'time': [],
            'voltage': [],
            'frequency': [],
            'battery_soc': [],
            'pv_output': [],
            'load_demand': [],
            'grid_power': []
        }
    if 'scenario_flags' not in st.session_state:
        st.session_state.scenario_flags = {
            'load_ramp': False,
            'pv_drop': False,
            'battery_disconnect': False,
            'grid_blackout': False,
            'peak_load': False
        }
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

# Microgrid simulation logic
def update_microgrid_state():
    if not st.session_state.simulation_running:
        return
    
    current_time = time.time()
    dt = current_time - st.session_state.last_update
    st.session_state.last_update = current_time
    
    # Base values with random fluctuations
    base_pv = 3.0 + random.uniform(-0.3, 0.3)
    base_load = 3.5 + random.uniform(-0.5, 0.5)
    
    # Apply scenario effects
    if st.session_state.scenario_flags['pv_drop']:
        base_pv *= 0.3  # 70% drop
    if st.session_state.scenario_flags['load_ramp']:
        base_load += 1.5  # Additional 1.5 kW
    if st.session_state.scenario_flags['peak_load']:
        base_load += 2.0  # Peak surge
    
    st.session_state.pv_output = max(0, base_pv)
    st.session_state.load_demand = max(0.5, base_load)
    
    # Battery management
    if not st.session_state.scenario_flags['battery_disconnect']:
        power_balance = st.session_state.pv_output - st.session_state.load_demand
        
        if power_balance > 0:  # Excess power - charge battery
            charge_power = min(power_balance, 2.0)  # Max 2kW charging
            if st.session_state.battery_soc < 95:
                st.session_state.battery_soc += (charge_power * dt / 3600) * 20  # Simplified SoC calculation
            power_balance -= charge_power
        elif power_balance < 0 and st.session_state.battery_soc > 10:  # Discharge battery
            discharge_power = min(abs(power_balance), 2.0)  # Max 2kW discharge
            st.session_state.battery_soc -= (discharge_power * dt / 3600) * 20
            power_balance += discharge_power
        
        st.session_state.battery_soc = max(0, min(100, st.session_state.battery_soc))
    
    # Grid interaction
    if st.session_state.scenario_flags['grid_blackout'] or st.session_state.island_mode:
        st.session_state.grid_power = 0
    else:
        st.session_state.grid_power = power_balance
    
    # Primary Control (Droop) - Frequency and Voltage regulation
    nominal_freq = 50.0
    nominal_voltage = 230.0
    
    # Frequency droop based on power imbalance
    power_imbalance = st.session_state.pv_output + abs(st.session_state.grid_power) - st.session_state.load_demand
    freq_deviation = -power_imbalance * 0.05  # Droop coefficient
    st.session_state.frequency = nominal_freq + freq_deviation + random.uniform(-0.1, 0.1)
    
    # Voltage regulation
    voltage_deviation = random.uniform(-3, 3)
    st.session_state.voltage = nominal_voltage + voltage_deviation
    
    # Secondary Control (ANN simulation)
    if st.session_state.secondary_ai_enabled:
        # Gradual correction towards nominal values
        freq_error = st.session_state.frequency - nominal_freq
        voltage_error = st.session_state.voltage - nominal_voltage
        
        if abs(freq_error) > 0.2:  # Threshold
            correction_factor = 0.2 * dt  # 5-second correction time
            st.session_state.frequency -= freq_error * correction_factor
        
        if abs(voltage_error) > 5:  # Threshold
            correction_factor = 0.2 * dt
            st.session_state.voltage -= voltage_error * correction_factor
    
    # Tertiary Control (RL simulation)
    if st.session_state.tertiary_ai_enabled:
        # Grid import optimization
        if st.session_state.grid_power > 1.0:  # Import threshold
            reduction_factor = 0.02 * dt  # 10-second optimization
            st.session_state.grid_power *= (1 - reduction_factor)
        
        # Battery SoC management
        if st.session_state.battery_soc < 20:  # Low SoC threshold
            # Prioritize charging by reducing load or increasing grid import
            if not st.session_state.island_mode:
                st.session_state.grid_power += 0.5 * dt
    
    # Clamp values to realistic ranges
    st.session_state.frequency = max(49.5, min(50.5, st.session_state.frequency))
    st.session_state.voltage = max(220, min(240, st.session_state.voltage))
    
    # Update data history
    current_timestamp = datetime.datetime.now()
    st.session_state.data_history['time'].append(current_timestamp)
    st.session_state.data_history['voltage'].append(st.session_state.voltage)
    st.session_state.data_history['frequency'].append(st.session_state.frequency)
    st.session_state.data_history['battery_soc'].append(st.session_state.battery_soc)
    st.session_state.data_history['pv_output'].append(st.session_state.pv_output)
    st.session_state.data_history['load_demand'].append(st.session_state.load_demand)
    st.session_state.data_history['grid_power'].append(st.session_state.grid_power)
    
    # Keep only last 50 data points
    for key in st.session_state.data_history:
        if len(st.session_state.data_history[key]) > 50:
            st.session_state.data_history[key] = st.session_state.data_history[key][-50:]

# Create charts
def create_charts():
    if len(st.session_state.data_history['time']) < 2:
        return None, None, None, None, None, None
    
    # Voltage Chart
    fig_voltage = go.Figure()
    fig_voltage.add_trace(go.Scatter(
        x=st.session_state.data_history['time'],
        y=st.session_state.data_history['voltage'],
        mode='lines',
        name='Gerilim',
        line=dict(color='#ff6b6b', width=2)
    ))
    fig_voltage.add_hline(y=230, line_dash="dash", line_color="gray", annotation_text="Nominal (230V)")
    fig_voltage.update_layout(
        title="AC Gerilim (V)",
        yaxis_title="Gerilim (V)",
        height=250,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Frequency Chart
    fig_frequency = go.Figure()
    fig_frequency.add_trace(go.Scatter(
        x=st.session_state.data_history['time'],
        y=st.session_state.data_history['frequency'],
        mode='lines',
        name='Frekans',
        line=dict(color='#4ecdc4', width=2)
    ))
    fig_frequency.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="Nominal (50Hz)")
    fig_frequency.update_layout(
        title="AC Frekans (Hz)",
        yaxis_title="Frekans (Hz)",
        height=250,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Battery SoC Chart
    fig_battery = go.Figure()
    fig_battery.add_trace(go.Scatter(
        x=st.session_state.data_history['time'],
        y=st.session_state.data_history['battery_soc'],
        mode='lines',
        name='Batarya SoC',
        line=dict(color='#45b7d1', width=2),
        fill='tonexty'
    ))
    fig_battery.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Düşük SoC (20%)")
    fig_battery.update_layout(
        title="Batarya Şarj Durumu (%)",
        yaxis_title="SoC (%)",
        height=250,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # PV Output Chart
    fig_pv = go.Figure()
    fig_pv.add_trace(go.Scatter(
        x=st.session_state.data_history['time'],
        y=st.session_state.data_history['pv_output'],
        mode='lines',
        name='PV Çıkış',
        line=dict(color='#f9ca24', width=2),
        fill='tozeroy'
    ))
    fig_pv.update_layout(
        title="PV Üretim (kW)",
        yaxis_title="Güç (kW)",
        height=250,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Load Demand Chart
    fig_load = go.Figure()
    fig_load.add_trace(go.Scatter(
        x=st.session_state.data_history['time'],
        y=st.session_state.data_history['load_demand'],
        mode='lines',
        name='Yük Talebi',
        line=dict(color='#e17055', width=2),
        fill='tozeroy'
    ))
    fig_load.update_layout(
        title="Yük Talebi (kW)",
        yaxis_title="Güç (kW)",
        height=250,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    # Grid Power Chart
    fig_grid = go.Figure()
    colors = ['red' if x < 0 else 'green' for x in st.session_state.data_history['grid_power']]
    fig_grid.add_trace(go.Scatter(
        x=st.session_state.data_history['time'],
        y=st.session_state.data_history['grid_power'],
        mode='lines',
        name='Şebeke Gücü',
        line=dict(color='#6c5ce7', width=2)
    ))
    fig_grid.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_grid.update_layout(
        title="Şebeke İthalat/İhracat (kW)",
        yaxis_title="Güç (kW)",
        height=250,
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig_voltage, fig_frequency, fig_battery, fig_pv, fig_load, fig_grid

# Main app
def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>⚡ AI Destekli AC Mikro Şebeke Kontrol Sistemi</h1>
        <p>Gerçek Zamanlı İzleme ve Akıllı Kontrol Paneli</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Kontrol Paneli")
        
        # Main simulation control
        st.markdown("#### Sistem Kontrolü")
        if st.button("🚀 Simülasyonu Başlat" if not st.session_state.simulation_running else "⏹️ Simülasyonu Durdur"):
            st.session_state.simulation_running = not st.session_state.simulation_running
        
        st.markdown("#### AI Kontrol Sistemleri")
        st.session_state.secondary_ai_enabled = st.checkbox(
            "🧠 İkincil AI (ANN)", 
            value=st.session_state.secondary_ai_enabled,
            help="Gerilim/frekans düzenlemesi için Yapay Sinir Ağı"
        )
        st.session_state.tertiary_ai_enabled = st.checkbox(
            "🤖 Üçüncül AI (RL)", 
            value=st.session_state.tertiary_ai_enabled,
            help="Ekonomik optimizasyon için Pekiştirmeli Öğrenme"
        )
        
        st.markdown("#### Çalışma Modu")
        st.session_state.island_mode = st.checkbox(
            "🏝️ Ada Modu", 
            value=st.session_state.island_mode,
            help="Ana şebekeden bağlantıyı kes"
        )
        
        st.markdown("#### Senaryo Testi")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Yük Artışı"):
                st.session_state.scenario_flags['load_ramp'] = not st.session_state.scenario_flags['load_ramp']
            
            if st.button("☀️ PV Düşüşü"):
                st.session_state.scenario_flags['pv_drop'] = not st.session_state.scenario_flags['pv_drop']
            
            if st.button("🔋 Batarya Kesintisi"):
                st.session_state.scenario_flags['battery_disconnect'] = not st.session_state.scenario_flags['battery_disconnect']
        
        with col2:
            if st.button("⚡ Şebeke Kesintisi"):
                st.session_state.scenario_flags['grid_blackout'] = not st.session_state.scenario_flags['grid_blackout']
            
            if st.button("🔥 Pik Yük"):
                st.session_state.scenario_flags['peak_load'] = not st.session_state.scenario_flags['peak_load']
        
        # Active scenarios display
        st.markdown("#### Aktif Senaryolar")
        scenario_names = {
            'load_ramp': 'Yük Artışı',
            'pv_drop': 'PV Düşüşü',
            'battery_disconnect': 'Batarya Kesintisi',
            'grid_blackout': 'Şebeke Kesintisi',
            'peak_load': 'Pik Yük'
        }
        for scenario, active in st.session_state.scenario_flags.items():
            if active:
                st.markdown(f"🔴 {scenario_names[scenario]}")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "status-online" if st.session_state.simulation_running else "status-offline"
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator {status_color}"></span>
            <strong>Sistem Durumu</strong><br>
            {'🟢 ÇEVRIMIÇI' if st.session_state.simulation_running else '🔴 ÇEVRIMDIŞI'}
        </div>
        """, unsafe_allow_html=True)
        st.metric("Gerilim", f"{st.session_state.voltage:.1f} V", f"{st.session_state.voltage-230:.1f}")
    
    with col2:
        ai_status = "🟢 AKTIF" if st.session_state.secondary_ai_enabled else "🔴 PASIF"
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator {'status-online' if st.session_state.secondary_ai_enabled else 'status-offline'}"></span>
            <strong>İkincil AI</strong><br>
            {ai_status}
        </div>
        """, unsafe_allow_html=True)
        st.metric("Frekans", f"{st.session_state.frequency:.2f} Hz", f"{st.session_state.frequency-50:.2f}")
    
    with col3:
        ai_status = "🟢 AKTIF" if st.session_state.tertiary_ai_enabled else "🔴 PASIF"
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator {'status-online' if st.session_state.tertiary_ai_enabled else 'status-offline'}"></span>
            <strong>Üçüncül AI</strong><br>
            {ai_status}
        </div>
        """, unsafe_allow_html=True)
        st.metric("Batarya SoC", f"{st.session_state.battery_soc:.1f} %", f"{st.session_state.battery_soc-80:.1f}")
    
    with col4:
        mode_status = "🏝️ ADA" if st.session_state.island_mode else "🔗 ŞEBEKE-BAĞLI"
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator {'status-warning' if st.session_state.island_mode else 'status-online'}"></span>
            <strong>Çalışma Modu</strong><br>
            {mode_status}
        </div>
        """, unsafe_allow_html=True)
        grid_direction = "İhracat" if st.session_state.grid_power < 0 else "İthalat"
        st.metric("Şebeke Gücü", f"{abs(st.session_state.grid_power):.2f} kW", grid_direction)
    
    # Charts section
    st.markdown("### 📊 Gerçek Zamanlı İzleme")
    
    # Update simulation state
    if st.session_state.simulation_running:
        update_microgrid_state()
    
    # Create and display charts
    charts = create_charts()
    if charts[0] is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.plotly_chart(charts[0], use_container_width=True)  # Voltage
            st.plotly_chart(charts[3], use_container_width=True)  # PV Output
        
        with col2:
            st.plotly_chart(charts[1], use_container_width=True)  # Frequency
            st.plotly_chart(charts[4], use_container_width=True)  # Load Demand
        
        with col3:
            st.plotly_chart(charts[2], use_container_width=True)  # Battery SoC
            st.plotly_chart(charts[5], use_container_width=True)  # Grid Power
    
    # Power flow summary
    st.markdown("### ⚡ Güç Akışı Özeti")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("PV Üretim", f"{st.session_state.pv_output:.2f} kW", "☀️")
    with col2:
        st.metric("Yük Tüketimi", f"{st.session_state.load_demand:.2f} kW", "🏭")
    with col3:
        battery_power = (st.session_state.pv_output - st.session_state.load_demand - st.session_state.grid_power)
        battery_status = "Şarj" if battery_power > 0 else "Deşarj" if battery_power < 0 else "Boşta"
        st.metric("Batarya Gücü", f"{abs(battery_power):.2f} kW", battery_status)
    with col4:
        grid_status = "İhracat" if st.session_state.grid_power < 0 else "İthalat" if st.session_state.grid_power > 0 else "Dengeli"
        st.metric("Şebeke Değişimi", f"{abs(st.session_state.grid_power):.2f} kW", grid_status)
    
    # Auto-refresh every 2 seconds when simulation is running
    if st.session_state.simulation_running:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
