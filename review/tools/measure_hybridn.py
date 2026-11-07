"""Read-only source/model investigation. Outputs only under review/tools/."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path.cwd()))
from hybrid.core.envelope import bounded_causal_envelope_db
from hybrid.core.nam_loader import load_nam
from hybrid.core.render import render
from hybrid.modes.blend import CrossoverConfig,blend,blend_weight
ROOT=Path('work/a2/AC30-6_TB_Brl_3_DI_and_Mesa_Boogie_Rev_D_Red_Koko_DI_Dynamic_Hybrid')
m=json.loads((ROOT/'training_manifest.json').read_text())
d=json.loads((ROOT/'hybrid_design.json').read_text())
export=Path.home() / 'Downloads' / 'AC30-6_TB_Brl_3_DI_and_Mesa_Boogie_Rev_D_Red_Koko_DI_Dynamic_Hybrid.nam'
cfg=CrossoverConfig(crossover_dbfs=d['crossover_dbfs'],transition_width_db=d['transition_width_db'],amp_b_trim_db=d['effective_b_trim_db'])
r={'export_sha256':hashlib.sha256(export.read_bytes()).hexdigest(),'manifest_export_sha256':m['training']['output_nam_sha256'],'baseline_head':'933c1218760d345a3b4396dedbedbe250852f21b','run_git_commit':m['git_commit']}
def db(v):return float(20*np.log10(max(float(v),1e-15)))
def coverage(x,sr):
 e=bounded_causal_envelope_db(x,sr);t=blend_weight(e,cfg);active=e>-60
 return {'active_seconds':float(active.sum()/sr),'envelope_percentiles_dbfs':np.percentile(e[active],[10,50,90,99]).tolist(),'mean_B_weight':float(t[active].mean()),'A_only_pct':float(100*np.mean(t[active]==0)),'B_only_pct':float(100*np.mean(t[active]==1)),'transition_pct':float(100*np.mean((t[active]>0)&(t[active]<1)))}
for name,p in [('training',ROOT/'input.wav'),('preview_raw',Path(m['preview_render_provenance']['source_paths']['di']))]:
 x,sr=sf.read(p,dtype='float32');r[name]=coverage(x,sr)
 if name=='preview_raw':
  r['preview_profiled']=coverage(x*10**(2.5/20),sr)
  energy=np.array([np.mean(x[i:i+sr]**2) for i in range(0,len(x)-sr,sr)])
  start=int(np.argmax(energy))*sr; clip=x[max(0,start-sr):start+4*sr];r['DI_excerpt_start_s']=max(0,start/sr-1)
models=[load_nam(d['amp_a_path']),load_nam(d['amp_b_path']),load_nam(export)]
r['DI_sweep']=[]
for gain in [-18,-12,-6,0,2.5,6]:
 x=(clip*10**(gain/20)).astype('float32');a=render(models[0],x,sr);b=render(models[1],x,sr)
 e=bounded_causal_envelope_db(x,sr);teacher,t=blend(e,a,b,cfg);teacher*=10**(m['output_gain']['applied_gain_db']/20)
 row={'gain_db':gain,'input_peak_dbfs':db(np.max(abs(x))),**coverage(x,sr),'variants':{}}
 for variant,slim in [('full',None),('lite',0.0)]:
  y=render(models[2],x,sr,slim=slim);mask=np.arange(len(x))>int(.2*sr)
  row['variants'][variant]={'esr':float(np.mean((y[mask]-teacher[mask])**2)/max(np.mean(teacher[mask]**2),1e-20)),'level_error_db':db(np.sqrt(np.mean(y[mask]**2))/max(np.sqrt(np.mean(teacher[mask]**2)),1e-15))}
 r['DI_sweep'].append(row);print('DI',json.dumps(row),flush=True)
# Coherent 100 Hz sine, .5 s per level, 0.2 s reset silence.
levels=[-48,-42,-36,-30,-24,-18,-12]
parts=[]
for level in levels:parts.extend([np.zeros(int(.2*sr)),10**(level/20)*np.sin(2*np.pi*100*np.arange(int(.5*sr))/sr)])
x=np.concatenate(parts).astype('float32');r['sine_sweep']={}
for label,model,slim in [('AC30',models[0],None),('Mesa',models[1],None),('full',models[2],None),('lite',models[2],0.0)]:
 y=render(model,x,sr,slim=slim);rows=[]
 for i,level in enumerate(levels):
  z=y[int(round((i*.7+.4)*sr)):int(round((i*.7+.7)*sr))].astype(float);n=len(z);time=np.arange(n)/sr
  amps=[float(2/n*abs(np.sum(z*np.exp(-2j*np.pi*100*h*time)))) for h in range(1,21)]
  rows.append({'input_sine_peak_dbfs':level,'output_rms_dbfs':db(np.sqrt(np.mean(z*z))),'H2_H20_over_H1_db':db(np.linalg.norm(amps[1:])/max(amps[0],1e-15))})
 r['sine_sweep'][label]=rows
Path('review/tools/hybridn_measurements.json').write_text(json.dumps(r,indent=2)+'\n')
print('DONE',flush=True)
