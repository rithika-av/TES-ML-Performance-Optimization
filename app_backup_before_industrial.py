from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Thermal Storage Intelligence', page_icon='♨️', layout='wide', initial_sidebar_state='expanded')
ROOT=Path(__file__).resolve().parent
SEARCH_ROOTS=[ROOT, ROOT.parent]
DATA_PATH=next((p/'dataset'/'TES_100000_requested_variables_only.csv' for p in SEARCH_ROOTS if (p/'dataset'/'TES_100000_requested_variables_only.csv').exists()),None)
TEMP_PATH=next((p/'dataset'/n for p in SEARCH_ROOTS for n in ['TES_PCM_Case1_dataset.csv','TES_PCM_Case2_temperature_dataset.csv'] if (p/'dataset'/n).exists()),None)

st.markdown('''<style>
:root{--bg:#061321;--panel:#0B1D30;--border:#174565;--cyan:#11D7E8;--blue:#139DFF;--teal:#13D6B2;--amber:#FFB52E;--text:#EAF4FF;--muted:#91B1D1}
html,body,[data-testid="stAppViewContainer"]{background:radial-gradient(ellipse at 70% -10%,#103458 0%,#071727 42%,#04101D 100%);color:var(--text)}
[data-testid="stHeader"]{background:rgba(4,16,29,.88)} .main .block-container{max-width:1800px;padding:1rem 1.4rem 2rem}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A2034,#071727);border-right:1px solid #174565} section[data-testid="stSidebar"] *{color:#DDEEFF}
section[data-testid="stSidebar"] [data-testid="stRadio"]>div{gap:7px} section[data-testid="stSidebar"] [data-testid="stRadio"] label{border:1px solid transparent;border-radius:9px;padding:9px;background:rgba(12,39,62,.42);font-weight:650}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{border-color:#17769A;background:rgba(17,215,232,.08)}
h1,h2,h3,h4{color:#F0F7FF!important;letter-spacing:-.3px} p,li,label,.stMarkdown{color:#C7DDF2}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(14,38,61,.97),rgba(7,24,41,.97));border:1px solid #1B4A6C;border-top:3px solid var(--cyan);border-radius:12px;padding:16px 18px;min-height:108px;box-shadow:0 8px 24px #0002}
[data-testid="stMetricLabel"]{color:#A8C9E8!important;font-weight:600}[data-testid="stMetricValue"]{color:#F1F8FF!important;font-weight:800}
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,div[data-baseweb="textarea"]{background:#091B2D;border-color:#285474;border-radius:8px;color:#F1F8FF}input,textarea{color:#F1F8FF!important}
.stButton>button,.stDownloadButton>button{background:linear-gradient(100deg,#087BAE,#087F8C);border:1px solid #17BBD2;border-radius:8px;color:white!important;font-weight:700}
[data-testid="stDataFrame"]{border:1px solid #174565;border-radius:10px;overflow:hidden}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;background:linear-gradient(100deg,#0D253CF5,#081B2EF5);border:1px solid #174565;border-radius:10px;padding:13px 18px;margin-bottom:22px}
.brand{font-size:19px;font-weight:850;letter-spacing:.5px;color:#F3F8FF}.brand-icon{color:#22DDF2;font-size:27px;margin-right:10px}.brand-sub{font-size:11px;letter-spacing:1.1px;color:#83B4DE;border-left:1px solid #315571;padding-left:16px;margin-left:15px}.online{border:1px solid #087E83;background:#00D2B414;color:#2BE8D1;padding:7px 13px;border-radius:22px;font-size:11px;font-weight:800;white-space:nowrap}
.page-title{display:flex;align-items:center;gap:14px;margin:4px 0 0}.page-title-icon{color:#16D7E8;font-size:31px}.subtitle{color:#6FC4FF;font-size:14px;margin:2px 0 18px 45px}
.panel{background:linear-gradient(145deg,#0B1F33F7,#071727FA);border:1px solid #174565;border-radius:10px;overflow:hidden;margin-bottom:15px}.panel-head{font-size:15px;font-weight:800;padding:13px 16px;border-bottom:1px solid #173B58;color:#EAF5FF}.panel-body{padding:14px 16px}
.op-row{display:flex;justify-content:space-between;gap:12px;padding:13px 2px;border-bottom:1px solid #183B57;font-size:13px}.op-row:last-child{border:0}.op-label{color:#A8C9E8}.op-value{color:#F3F8FF;font-size:15px;font-weight:800;text-align:right}.state-pill{border:1px solid #00BBAA;color:#18E2CB;background:#00CDB414;border-radius:20px;padding:5px 12px;font-size:11px;font-weight:800}
.rec{border:1px solid #1A4969;border-radius:9px;background:#0C253CB8;padding:12px;display:flex;gap:12px;margin-bottom:10px}.rec-num{min-width:35px;height:35px;border:1px solid #087CA7;background:#103653;color:#1ED5F1;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:18px;font-weight:850}.rec-title{font-size:12px;font-weight:800;color:#EAF5FF;margin-bottom:5px}.rec-text{font-size:11px;line-height:1.55;color:#9FC1E1}.note{color:#FFBD52;font-size:11px;padding:6px 0}.muted{color:#83A9CA;font-size:12px}.progress-label{display:flex;justify-content:space-between;font-size:11px;color:#B9D1E9;margin:12px 0 5px}.track{height:9px;border-radius:8px;background:#18344E;overflow:hidden}.fill{height:100%;border-radius:8px}.footer{color:#7599B9;font-size:11px;margin-top:8px}hr{border-color:#193A55}
</style>''',unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_csv(path):
    try:return pd.read_csv(path,low_memory=False)
    except Exception:return pd.DataFrame()

dataset=read_csv(str(DATA_PATH)) if DATA_PATH else pd.DataFrame()
tempdata=read_csv(str(TEMP_PATH)) if TEMP_PATH else pd.DataFrame()

def curve_for(inlet=320,initial=260,hours=10,flow=.2,n=101):
    t=np.linspace(0,hours,n); rate=.30+min(max(flow,.01),1)*.13
    measured=inlet-(inlet-initial)*np.exp(-rate*t)
    forecast=inlet+5-(inlet+5-(initial+6))*np.exp(-rate*.92*t)
    return pd.DataFrame({'Time (h)':t,'Measured / reference (illustrative)':measured,'Forecast (illustrative)':forecast})

def chart(df,x,ys,xt,yt,height=320):
    fig=go.Figure(); colors=['#12D6C7','#FFB72E','#168FFF','#B18CFF']
    for i,y in enumerate(ys):
        if y in df:fig.add_trace(go.Scatter(x=df[x],y=df[y],mode='lines',name=y,line=dict(color=colors[i%len(colors)],width=2.7)))
    fig.update_layout(height=height,margin=dict(l=8,r=8,t=8,b=8),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(4,18,32,.45)',font=dict(color='#BCD4EC',size=11),legend=dict(orientation='h',yanchor='bottom',y=-.22,x=0,font=dict(size=10)),xaxis=dict(title=xt,gridcolor='#173955',linecolor='#315A78',zeroline=False),yaxis=dict(title=yt,gridcolor='#173955',linecolor='#315A78',zeroline=False),hovermode='x unified')
    return fig

def start_panel(title,icon='▦'):
    st.markdown(f'<div class="panel"><div class="panel-head">{icon}&nbsp;&nbsp; {title}</div><div class="panel-body">',unsafe_allow_html=True)
def end_panel():st.markdown('</div></div>',unsafe_allow_html=True)
def progress(label,pct,color,right):
    st.markdown(f'<div class="progress-label"><span>{label}</span><span>{right}</span></div><div class="track"><div class="fill" style="width:{np.clip(pct,0,100)}%;background:{color};"></div></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="text-align:center;padding:15px 4px 19px"><div style="font-size:36px;color:#19D9EE">▤</div><div style="font-size:13px;font-weight:900;color:#F0F7FF">THERMAL STORAGE</div><div style="font-size:9px;letter-spacing:1.7px;color:#75B5E5;margin-top:4px">INTELLIGENCE PLATFORM</div></div>',unsafe_allow_html=True)
    pages=['Plant Overview','Live Performance','Thermal Simulation','ML Forecasts','Optimization','Asset Health','Reports']
    page=st.radio('NAVIGATION',pages,index=0,label_visibility='collapsed',key='nav_page')
    st.markdown('---');st.markdown('**SYSTEM STATUS**')
    st.markdown('<div style="color:#16DCC1;font-size:12px;margin:12px 0">●&nbsp; Interface online</div><div class="muted">Plant data connection: not configured</div>',unsafe_allow_html=True)
    if DATA_PATH:st.caption('Local dataset detected: '+DATA_PATH.name)
    else:st.markdown('<div style="color:#F5B94C;font-size:11px;margin-top:12px">● Demo display mode</div>',unsafe_allow_html=True)

st.markdown(f'<div class="topbar"><div style="display:flex;align-items:center;flex-wrap:wrap"><span class="brand-icon">▤</span><span class="brand">THERMAL STORAGE INTELLIGENCE</span><span class="brand-sub">PLANT PERFORMANCE CONSOLE</span></div><span class="online">● &nbsp; SYSTEM ONLINE</span><span style="font-size:11px;color:#A9C9E5;border:1px solid #214968;border-radius:8px;padding:8px 11px">▦ &nbsp; {date.today().strftime("%b %d, %Y")} &nbsp;⌄</span></div>',unsafe_allow_html=True)
curve=curve_for()

if page=='Plant Overview':
    st.markdown('<div class="page-title"><span class="page-title-icon">▤</span><h1 style="margin:0;font-size:29px">Thermal Storage Performance</h1></div><div class="subtitle">Operational intelligence &nbsp;•&nbsp; Charging / discharging &nbsp;•&nbsp; Thermal efficiency</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric('Thermal Efficiency','91.8%','+2.3% vs previous period');b.metric('Stored Energy','4.62 MWh','+0.48 MWh vs previous period');c.metric('State of Charge','78.4%','+6.1% vs previous period');d.metric('Forecast Confidence','96.2%','+1.8% vs previous period')
    l,r=st.columns([1.85,1])
    with l:
        start_panel('Thermal Response — Measured vs Forecast','⌁');st.plotly_chart(chart(curve,'Time (h)',list(curve.columns[1:]),'Time (h)','Temperature (K)',310),use_container_width=True,config={'displayModeBar':False});end_panel()
    with r:
        start_panel('Operating Envelope','◉')
        for label,value in [('Inlet Temperature','320 K'),('Mass Flow Rate','0.20 kg/s'),('PCM Core Temperature','301.4 K')]:st.markdown(f'<div class="op-row"><span class="op-label">{label}</span><span class="op-value">{value}</span></div>',unsafe_allow_html=True)
        st.markdown('<div class="op-row"><span class="op-label">Operating State</span><span class="state-pill">CHARGING</span></div>',unsafe_allow_html=True);end_panel()
    l,r=st.columns([1.45,1])
    with l:
        start_panel('Charge Cycle Summary','▣');x,y=st.columns([.9,1.1])
        with x:
            progress('Energy Stored',77,'#12D6C7','4.62 / 6.00 MWh • 77%');progress('Charging Progress',78.4,'#169DFF','78%');progress('Current Temperature (avg)',84,'#FFB347','301.4 K');progress('Cycle Duration (est.)',48,'#86B5E7','2.4 h')
        with y:
            st.markdown('''<div style="height:225px;display:flex;align-items:center"><svg viewBox="0 0 440 220" style="width:100%"><defs><linearGradient id="tank" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="#FFB13B"/><stop offset=".48" stop-color="#177EAD"/><stop offset="1" stop-color="#0D3154"/></linearGradient></defs><path d="M8 58 H110" stroke="#FFB52E" stroke-width="4"/><path d="M100 50 L113 58 L100 66" fill="#FFB52E"/><text x="8" y="34" fill="#FFBE49" font-size="13">Hot inlet fluid</text><text x="8" y="49" fill="#FFBE49" font-size="12">320 K</text><rect x="120" y="42" width="88" height="108" rx="9" fill="#164668" stroke="#80C9F7" stroke-width="2"/><path d="M130 55 l12 12 12 -12 12 12 12 -12 12 12 12 -12 M130 80 l12 12 12 -12 12 12 12 -12 12 12 12 -12 M130 105 l12 12 12 -12 12 12 12 -12 12 12 12 -12 M130 130 l12 10 12 -10 12 10 12 -10 12 10 12 -10" fill="none" stroke="#BFE8FF" stroke-width="2"/><path d="M208 66 H273" stroke="#B9D9EF" stroke-width="4"/><path d="M265 58 L278 66 L265 74" fill="#B9D9EF"/><path d="M208 135 H245 Q258 135 258 150 V168 H286" stroke="#17B8F0" stroke-width="4" fill="none"/><path d="M278 160 L291 168 L278 176" fill="#17B8F0"/><path d="M295 32 Q340 5 385 32 L385 154 Q385 181 340 185 Q295 181 295 154 Z" fill="url(#tank)" stroke="#B9D9EF" stroke-width="2"/><ellipse cx="340" cy="32" rx="45" ry="14" fill="#163F60" stroke="#B9D9EF" stroke-width="2"/><text x="340" y="91" text-anchor="middle" fill="white" font-size="14">PCM</text><text x="340" y="110" text-anchor="middle" fill="white" font-size="13">301.4 K</text><text x="164" y="173" text-anchor="middle" fill="#D7EAFB" font-size="12">Heat exchanger</text><text x="340" y="205" text-anchor="middle" fill="#D7EAFB" font-size="12">PCM storage tank</text><path d="M385 124 H425" stroke="#17B8F0" stroke-width="3"/><path d="M416 116 L429 124 L416 132" fill="#17B8F0"/><text x="388" y="91" fill="#31D8F3" font-size="11">Cooled</text><text x="388" y="105" fill="#31D8F3" font-size="11">outlet</text></svg></div>''',unsafe_allow_html=True)
        end_panel()
    with r:
        start_panel('Optimization Recommendations','♧')
        st.markdown('<div class="rec"><div class="rec-num">1</div><div><div class="rec-title">Evaluate flow setpoint within validated operating limits</div><div class="rec-text">Current flow rate (0.20 kg/s) is a reference. Test nearby setpoints only within equipment and safety limits.</div></div></div><div class="rec"><div class="rec-num">2</div><div><div class="rec-title">Forecast indicates charge completion in ~2.4 h</div><div class="rec-text">Illustrative dashboard estimate. Replace with validated model output before operational decisions.</div></div></div>',unsafe_allow_html=True);end_panel()
    st.markdown('<div class="note">⚠ &nbsp; Illustrative interface • Connect validated plant data and verified ML outputs before operational use.</div>',unsafe_allow_html=True)

elif page=='Live Performance':
    st.markdown('<div class="page-title"><span class="page-title-icon">⌁</span><h1 style="margin:0;font-size:29px">Live Performance</h1></div><div class="subtitle">Operating envelope, thermal response and trend inspection</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4);a.metric('Inlet Temperature','320 K');b.metric('Mass Flow Rate','0.20 kg/s');c.metric('PCM Core Temperature','301.4 K');d.metric('Operating State','CHARGING')
    l,r=st.columns([1.5,1])
    with r:
        start_panel('Live Setpoints','⚙');inlet=st.slider('Inlet temperature (K)',280,380,320);flow=st.slider('Mass flow rate (kg/s)',.01,.5,.20,.01);initial=st.slider('Initial/reference temperature (K)',250,320,260)
        if st.button('Update trend',use_container_width=True):st.session_state.live_curve=curve_for(inlet,initial,10,flow);st.success('Illustrative trend refreshed.')
        st.caption('Demo controls only — no sensors or PLC are connected.');end_panel()
    with l:
        start_panel('Temperature Trend','⌁');st.plotly_chart(chart(st.session_state.get('live_curve',curve),'Time (h)',list(curve.columns[1:]),'Time (h)','Temperature (K)',390),use_container_width=True,config={'displayModeBar':False});end_panel()

elif page=='Thermal Simulation':
    st.markdown('<div class="page-title"><span class="page-title-icon">⚗</span><h1 style="margin:0;font-size:29px">Thermal Simulation</h1></div><div class="subtitle">Configure reference conditions and inspect the thermal response</div>',unsafe_allow_html=True)
    st.info('This screen calculates an illustrative first-order response; it does not launch OpenFOAM or a validated physics solver.')
    a,b,c=st.columns(3)
    with a: inlet=st.number_input('Inlet temperature (K)',250.,500.,320.,1.)
    with b: initial=st.number_input('Initial PCM temperature (K)',200.,400.,295.,1.)
    with c: flow=st.number_input('Mass flow rate (kg/s)',.01,2.,.20,.01)
    a,b,c=st.columns(3)
    with a: duration=st.slider('Duration (hours)',1,24,10)
    with b: mass=st.number_input('PCM mass (kg)',.1,10000.,80.,5.)
    with c: cp=st.number_input('PCM specific heat (J/kg·K)',100.,5000.,2200.,50.)
    sim=curve_for(inlet,initial,float(duration),flow,121);energy=max(0,mass*cp*(float(sim.iloc[-1,1])-initial)/3_600_000)
    if st.button('Run illustrative simulation',type='primary'):st.session_state.sim=sim;st.session_state.energy=energy
    sim=st.session_state.get('sim',sim);a,b,c=st.columns(3);a.metric('Final reference temperature',f'{sim.iloc[-1,1]:.1f} K');b.metric('Sensible energy estimate',f'{st.session_state.get("energy",energy):.3f} kWh');c.metric('Simulation horizon',f'{duration} h')
    start_panel('Calculated Thermal Response','⌁');st.plotly_chart(chart(sim,'Time (h)',list(sim.columns[1:]),'Time (h)','Temperature (K)',390),use_container_width=True,config={'displayModeBar':False});end_panel()

elif page=='ML Forecasts':
    st.markdown('<div class="page-title"><span class="page-title-icon">⌁</span><h1 style="margin:0;font-size:29px">ML Forecasts</h1></div><div class="subtitle">Dataset inspection and model integration workspace</div>',unsafe_allow_html=True)
    data=dataset if not dataset.empty else tempdata
    path=DATA_PATH if not dataset.empty else TEMP_PATH
    if not data.empty:
        st.success(f'Local dataset loaded: {path.name} • {len(data):,} rows • {len(data.columns)} columns')
        numeric=data.select_dtypes(include=np.number)
        if not numeric.empty:
            col=st.selectbox('Inspect numeric variable',list(numeric.columns));values=numeric[col].dropna().iloc[:1000].reset_index(drop=True)
            df=pd.DataFrame({'Sample':np.arange(len(values)),'Value':values});st.plotly_chart(chart(df,'Sample',['Value'],'Sample index',str(col),330),use_container_width=True,config={'displayModeBar':False})
        st.dataframe(data.head(25),use_container_width=True)
        st.download_button('Download dataset sample CSV',data.head(5000).to_csv(index=False).encode(), 'tes_dataset_sample.csv','text/csv')
    else:st.warning('Dataset not found. Put your CSV in the project dataset/ folder.')
    st.warning('A trained ML model is not loaded by this interface. Connect your evaluated model and scaler before describing forecasts as ML predictions.')

elif page=='Optimization':
    st.markdown('<div class="page-title"><span class="page-title-icon">⚙</span><h1 style="margin:0;font-size:29px">Optimization</h1></div><div class="subtitle">Explore candidate setpoints within user-defined bounds</div>',unsafe_allow_html=True)
    st.warning('Sensitivity sweep is illustrative, not an engineering recommendation or equipment control. Use validated safety limits.')
    a,b,c=st.columns(3)
    with a:low=st.number_input('Minimum flow (kg/s)',.01,1.,.10,.01)
    with b:high=st.number_input('Maximum flow (kg/s)',.02,2.,.30,.01)
    with c:inlet=st.number_input('Inlet temperature (K)',280.,450.,320.,1.)
    if high<=low:st.error('Maximum flow must be greater than minimum flow.')
    else:
        flows=np.linspace(low,high,25);score=np.clip(100*(1-np.exp(-flows*5))*(1-.0015*max(inlet-320,0)),0,100);out=pd.DataFrame({'Flow rate (kg/s)':flows,'Illustrative relative score':score});best=out.iloc[int(np.argmax(score))]
        a,b=st.columns(2);a.metric('Highest score in sweep',f'{best.iloc[1]:.1f}/100');b.metric('Candidate flow',f'{best.iloc[0]:.3f} kg/s')
        start_panel('Sensitivity Sweep','⌁');st.plotly_chart(chart(out,out.columns[0],[out.columns[1]],'Flow rate (kg/s)','Relative score (not validated)',350),use_container_width=True,config={'displayModeBar':False});end_panel();st.dataframe(out.round(4),use_container_width=True);st.download_button('Export sweep CSV',out.to_csv(index=False).encode(),'tes_optimization_sweep.csv','text/csv')

elif page=='Asset Health':
    st.markdown('<div class="page-title"><span class="page-title-icon">⬡</span><h1 style="margin:0;font-size:29px">Asset Health</h1></div><div class="subtitle">Manual readiness checklist and operating-system status</div>',unsafe_allow_html=True)
    st.info('No sensor feeds or maintenance system are connected. Checklist state is local to this dashboard.')
    a,b,c=st.columns(3);a.metric('Assets configured','1','TES placeholder');b.metric('Sensor connection','Offline','Not configured');c.metric('Maintenance alerts','Not assessed','No CMMS connection')
    start_panel('Manual readiness checklist','☑')
    for item in ['Inlet and outlet temperature sensors calibrated','Mass-flow meter calibration verified','Pressure-drop limits documented','PCM containment inspection complete','Safety interlocks tested by authorized personnel','Model outputs validated against measurements']:st.checkbox(item,key='health_'+item)
    end_panel()
    if st.button('Summarize checklist'):
        items=['Inlet and outlet temperature sensors calibrated','Mass-flow meter calibration verified','Pressure-drop limits documented','PCM containment inspection complete','Safety interlocks tested by authorized personnel','Model outputs validated against measurements'];done=sum(st.session_state.get('health_'+x,False) for x in items);st.write(f'Checklist progress: {done} of {len(items)} items checked. This is not a safety certification.')

elif page=='Reports':
    st.markdown('<div class="page-title"><span class="page-title-icon">▤</span><h1 style="margin:0;font-size:29px">Reports</h1></div><div class="subtitle">Export reference trends and local project datasets</div>',unsafe_allow_html=True)
    report=curve.copy();report['Inlet temperature reference (K)']=320.;report['Mass flow reference (kg/s)']=.20;report['Operating mode']='Illustrative charging'
    a,b=st.columns(2)
    with a:
        start_panel('Thermal response report','▤');st.dataframe(report.head(20),use_container_width=True);st.download_button('Download thermal response CSV',report.to_csv(index=False).encode(),'tes_thermal_response_report.csv','text/csv',use_container_width=True);end_panel()
    with b:
        start_panel('Project data files','▧')
        if DATA_PATH:
            st.write(f'Main dataset: `{DATA_PATH}`');st.write(f'{len(dataset):,} rows • {len(dataset.columns)} columns');st.download_button('Download main dataset',dataset.to_csv(index=False).encode(),'tes_dataset.csv','text/csv',use_container_width=True)
        else:st.write('Main dataset not found in the expected dataset/ directory.')
        if TEMP_PATH and not tempdata.empty:
            st.write(f'Temperature data: `{TEMP_PATH.name}`');st.write(f'{len(tempdata):,} rows • {len(tempdata.columns)} columns');st.download_button('Download temperature data',tempdata.to_csv(index=False).encode(),'tes_temperature_data.csv','text/csv',use_container_width=True)
        else:st.caption('No separate temperature CSV found.')
        end_panel()

st.markdown('<div class="footer">THERMAL STORAGE INTELLIGENCE • Local research dashboard • Validate measurements, models and operating limits before industrial deployment.</div>',unsafe_allow_html=True)
