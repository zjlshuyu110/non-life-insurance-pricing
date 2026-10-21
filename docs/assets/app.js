/* All browser-side data are aggregates or predictions on an illustrative grid. */
(() => {
  'use strict';
  const D = window.STUDY_DATA, M = window.LabMath, rows = M.decode(D);
  const $ = id => document.getElementById(id);
  const C = {ink:'#e7eff2', muted:'#90a6b1', teal:'#45d6c0', coral:'#ed966a', gold:'#cfb57a', grid:'#233945', panel:'#0c1e2b'};
  const state = {model:'fs', age:'all', region:'all', vehicle:'all', cap:'200000', view:'surface', profileAge:40, bonus:50};
  let renderedSurfaceView = null;
  const names = {fs:'Poisson × Gamma', tw:'Tweedie GLM', base:'Portfolio mean'};
  const whole = M.aggregate(rows);
  const nf = (v, d=0) => v == null || !Number.isFinite(v) ? '—' : v.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
  const eur = (v,d=0) => v == null ? '—' : '€'+nf(v,d);
  const axis = title => ({title:{text:title,font:{size:10,color:C.muted}},gridcolor:C.grid,zeroline:false,tickfont:{size:10,color:C.muted},linecolor:C.grid,automargin:true});
  const base = () => ({paper_bgcolor:C.panel,plot_bgcolor:C.panel,font:{family:'Segoe UI, Arial, sans-serif',color:C.ink,size:11},margin:{l:59,r:25,t:25,b:66},legend:{orientation:'h',y:-.23,x:0,font:{size:10}},hoverlabel:{bgcolor:'#173343',font:{color:C.ink,size:12}},hovermode:'x unified',transition:{duration:0}});
  const config = {responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d'],toImageButtonOptions:{format:'png',scale:2}};
  function plot(id, traces, layout={}) { return Plotly.react($(id), traces, {...base(),...layout},config); }
  function activate(selector, attr, value) { document.querySelectorAll(selector).forEach(b => {const on=b.dataset[attr]===value;b.classList.toggle('active',on);b.setAttribute('aria-pressed',String(on));}); }
  for (const [id, labels] of [['filter-age',D.ageLabels],['filter-region',D.regions],['filter-vehicle',D.vehicleLabels]]) labels.forEach((label,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=label;$(id).append(option);});
  $('profile-details').textContent=Object.entries(D.surface.profile).filter(([k])=>!['DrivAge','BonusMalus'].includes(k)).map(([k,v])=>k+': '+v).join(' · ')+'. Values selected from the training portfolio.';

  function renderSurface() {
    // WebGL scenes and Cartesian axes cannot safely share Plotly's retained state.
    if (renderedSurfaceView !== state.view) {
      Plotly.purge($('surface-chart'));
      renderedSurfaceView = state.view;
    }
    const surface=D.surface, z=surface.values[state.model], a=surface.ages.indexOf(state.profileAge), b=surface.bonuses.indexOf(state.bonus);
    $('age-output').value=state.profileAge;$('bonus-output').value=state.bonus;
    $('profile-price').textContent=eur(z[a][b]);$('profile-model').textContent=names[state.model];
    activate('#surface-switch button','view',state.view);
    document.querySelector('.chart-hint').textContent=state.view==='surface'?'DRAG TO ROTATE · SCROLL TO ZOOM':'DRIVER AGE × BONUS–MALUS · SELECT A PROFILE BELOW';
    const scale=[[0,'#143e46'],[.25,'#1e7778'],[.5,'#45c7b1'],[.75,'#ccb77c'],[1,'#ed966a']];
    if(state.view==='heatmap') {
      plot('surface-chart',[{type:'heatmap',x:surface.bonuses,y:surface.ages,z,colorscale:scale,colorbar:{title:{text:'EUR',side:'top'},thickness:9,tickfont:{size:9}},hovertemplate:'Age %{y}<br>Bonus–malus %{x}<br>Pure premium €%{z:,.0f}<extra></extra>'},{type:'scatter',x:[state.bonus],y:[state.profileAge],mode:'markers',marker:{color:C.ink,size:9,line:{color:C.gold,width:2}},showlegend:false,hoverinfo:'skip'}],{margin:{l:48,r:55,t:16,b:40},xaxis:axis('Bonus–malus'),yaxis:axis('Driver age'),hovermode:'closest'});
      return;
    }
    const sceneAxis=title=>({title:{text:title,font:{size:10,color:C.muted}},tickfont:{size:8,color:C.muted},gridcolor:C.grid,zeroline:false,showbackground:false});
    plot('surface-chart',[{type:'surface',x:surface.bonuses,y:surface.ages,z,colorscale:scale,showscale:false,opacity:.95,contours:{z:{show:true,usecolormap:true,highlightcolor:C.ink,project:{z:false}}},hovertemplate:'Age %{y}<br>Bonus–malus %{x}<br>Pure premium €%{z:,.0f}<extra></extra>'},{type:'scatter3d',x:[state.bonus],y:[state.profileAge],z:[z[a][b]*1.01],mode:'markers',marker:{color:C.gold,size:5,line:{color:'#fff',width:1}},showlegend:false,hovertemplate:'Selected profile<br>€%{z:,.0f} / year<extra></extra>'}],{margin:{l:0,r:0,t:0,b:0},uirevision:'retain-camera',hovermode:'closest',scene:{xaxis:sceneAxis('Bonus–malus'),yaxis:sceneAxis('Driver age'),zaxis:sceneAxis('EUR / year'),camera:{eye:{x:1.4,y:-1.65,z:.95}},aspectratio:{x:1.25,y:1,z:.7},bgcolor:C.panel}}).catch(()=>{state.view='heatmap';renderSurface();});
  }

  function renderCohort() {
    const selected=M.filter(rows,state), total=M.aggregate(selected), expected=total[state.model], ratio=M.ratio(expected,total.loss);
    const labels=[state.age==='all'?'All ages':D.ageLabels[+state.age],state.region==='all'?'All regions':D.regions[+state.region],state.vehicle==='all'?'All vehicles':D.vehicleLabels[+state.vehicle]];
    $('cohort-status').textContent=labels.join(' / ')+' · '+nf(100*total.policies/whole.policies,1)+'% of held-out policies'+(total.exposure<100?' · Small exposure: interpret cautiously.':'');
    $('kpi-policies').textContent=nf(total.policies);$('kpi-exposure').textContent=nf(total.exposure)+' exposure-years';
    $('kpi-actual').textContent=eur(M.ratio(total.loss,total.exposure));$('kpi-predicted').textContent=eur(M.ratio(expected,total.exposure));
    $('kpi-model').textContent=names[state.model];$('kpi-ratio').textContent=ratio==null?'—':nf(ratio*100,1)+'%';
    $('kpi-gap').textContent=ratio==null?'No observed losses in selection':nf(Math.abs(ratio-1)*100,1)+'% '+(ratio>=1?'above':'below')+' observed losses';
    $('kpi-ratio').style.color=ratio!=null&&Math.abs(ratio-1)>.1?C.coral:C.teal;
    const deciles=M.group(selected,'decile',10,1);
    plot('decile-chart',[{type:'bar',x:deciles.map(g=>g.group),y:deciles.map(g=>M.ratio(g.loss,g.exposure)),name:'Observed',marker:{color:'#536f7d',line:{width:0}},hovertemplate:'Decile %{x}<br>Observed €%{y:,.0f}<extra></extra>'},{type:'scatter',x:deciles.map(g=>g.group),y:deciles.map(g=>M.ratio(g[state.model],g.exposure)),name:names[state.model],mode:'lines+markers',connectgaps:false,line:{color:C.teal,width:2.5},marker:{size:6},customdata:deciles.map(g=>[g.policies,g.exposure]),hovertemplate:'Expected €%{y:,.0f}<br>%{customdata[0]:,} policies<br>%{customdata[1]:,.0f} exposure-years<extra></extra>'}],{xaxis:{...axis('Fixed risk decile'),dtick:1},yaxis:axis('EUR / exposure-year'),bargap:.42});
    const models=['base','tw','fs'];
    plot('duel-chart',[{type:'bar',orientation:'h',x:models.map(k=>M.ratio(total['dev_'+k],total.exposure)),y:models.map(k=>names[k]),marker:{color:models.map(k=>k===state.model?C.teal:k==='base'?'#536f7d':C.gold)},text:models.map(k=>nf(M.ratio(total['dev_'+k],total.exposure),3)),textposition:'outside',cliponaxis:false,hovertemplate:'%{y}<br>Deviance %{x:.4f}<extra></extra>'}],{margin:{l:112,r:64,t:38,b:53},xaxis:{...axis('Weighted deviance · lower is better'),rangemode:'tozero'},yaxis:{...axis(''),showgrid:false},showlegend:false,bargap:.52,hovermode:'closest'});
    const byAge=M.group(selected,'age',D.ageLabels.length);
    plot('age-chart',[{type:'scatter',x:D.ageLabels,y:byAge.map(g=>M.ratio(g.loss,g.exposure)),name:'Observed',mode:'lines+markers',line:{color:C.gold,width:2},marker:{size:7}},{type:'scatter',x:D.ageLabels,y:byAge.map(g=>M.ratio(g[state.model],g.exposure)),name:names[state.model],mode:'lines+markers',line:{color:C.teal,width:2.5},marker:{size:7}}],{xaxis:axis('Driver age'),yaxis:axis('EUR / exposure-year'),margin:{l:59,r:25,t:42,b:90}});
    renderMatrix();
  }

  function renderMatrix() {
    const overview=M.filter(rows,{age:'all',region:'all',vehicle:state.vehicle});
    const z=[],details=[];
    D.regions.forEach((region,r)=>{const line=[],notes=[];D.ageLabels.forEach((age,a)=>{const g=M.aggregate(overview.filter(row=>row.region===r&&row.age===a));line.push(g.exposure>=30?M.ratio(g.loss,g[state.model]):null);notes.push([g.exposure,g.policies]);});z.push(line);details.push(notes);});
    plot('matrix-chart',[{type:'heatmap',x:D.ageLabels,y:D.regions,z,zmin:0,zmax:2,zmid:1,customdata:details,colorscale:[[0,'#237a78'],[.5,'#17303e'],[1,'#ed966a']],colorbar:{title:{text:'A / E',side:'top'},thickness:8,len:.8,tickfont:{size:9}},hoverongaps:false,hovertemplate:'%{y} · %{x}<br>Observed / expected: %{z:.2f}<br>%{customdata[0]:,.0f} exposure-years<br>%{customdata[1]:,} policies<extra></extra>'}],{margin:{l:49,r:55,t:17,b:53},xaxis:{...axis('Driver age'),tickangle:-25},yaxis:{...axis(''),dtick:1,tickfont:{size:9,color:C.muted},autorange:'reversed'},hovermode:'closest'}).then(()=>{const el=$('matrix-chart');el.removeAllListeners('plotly_click');el.on('plotly_click',event=>{const p=event.points[0];if(p.z==null)return;state.age=String(D.ageLabels.indexOf(p.x));state.region=String(D.regions.indexOf(p.y));$('filter-age').value=state.age;$('filter-region').value=state.region;renderCohort();});});
  }

  function renderTail() {
    const modelKey=state.model==='fs'?'FrequencySeverity':'Tweedie';
    const scenario=D.sensitivity.filter(r=>String(r.loss_cap)===state.cap);
    const row=scenario.find(r=>r.model===modelKey), baseline=scenario.find(r=>r.model==='Baseline');
    $('tail-ratio').textContent=nf(row.predicted_actual*100,1)+'%';$('tail-ratio').style.color=state.cap==='Uncapped'?C.coral:C.teal;
    $('tail-actual').textContent=eur(row.actual_loss);$('tail-predicted').textContent=eur(row.predicted_loss);
    $('tail-title').textContent=state.cap==='Uncapped'?'A few claims can move the portfolio.':state.cap==='50000'?'A tighter cap, a different target.':'The reference case, in perspective.';
    $('tail-narrative').textContent=state.cap==='Uncapped'?'With no loss cap, '+names[state.model]+' predicts '+nf((row.predicted_actual-1)*100,1)+'% more than observed. The portfolio-mean baseline also overpredicts by '+nf((baseline.predicted_actual-1)*100,1)+'%. This points to sensitivity to the train/test loss mix, not just model choice.':names[state.model]+' predicts '+nf(Math.abs(row.predicted_actual-1)*100,1)+'% '+(row.predicted_actual>=1?'above':'below')+' observed losses at this cap. Capping changes the target and excludes part of the tail; closer calibration does not make the missing tail disappear.';
    plot('tail-chart',[{type:'bar',x:scenario.map(r=>({Baseline:'Portfolio mean',FrequencySeverity:'Poisson × Gamma',Tweedie:'Tweedie GLM'}[r.model])),y:scenario.map(r=>100*r.predicted_actual),marker:{color:scenario.map(r=>r.model===modelKey?C.teal:r.model==='Baseline'?'#536f7d':C.gold)},text:scenario.map(r=>nf(r.predicted_actual*100,1)+'%'),textposition:'outside',cliponaxis:false,hovertemplate:'%{x}<br>Expected / observed %{y:.2f}%<extra></extra>'}],{margin:{l:58,r:25,t:48,b:58},xaxis:{...axis(''),showgrid:false},yaxis:{...axis('Predicted / observed (%)'),range:[0,140]},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:100,y1:100,line:{color:C.coral,width:1,dash:'dot'}}],annotations:[{xref:'paper',x:1,y:100,yshift:10,text:'100%',showarrow:false,font:{size:9,color:C.coral},xanchor:'right'}],bargap:.5,showlegend:false,hovermode:'closest'});
  }

  document.querySelectorAll('.model-switch button').forEach(button=>button.addEventListener('click',()=>{state.model=button.dataset.model;activate('.model-switch button','model',state.model);renderSurface();renderCohort();renderTail();}));
  for(const key of ['age','region','vehicle'])$('filter-'+key).addEventListener('change',e=>{state[key]=e.target.value;renderCohort();});
  $('reset-filters').addEventListener('click',()=>{for(const key of ['age','region','vehicle']){state[key]='all';$('filter-'+key).value='all';}renderCohort();});
  for(const [id,key] of [['profile-age','profileAge'],['profile-bonus','bonus']])$(id).addEventListener('input',e=>{state[key]=Number(e.target.value);renderSurface();});
  document.querySelectorAll('#surface-switch button').forEach(button=>button.addEventListener('click',()=>{state.view=button.dataset.view;renderSurface();}));
  document.querySelectorAll('#cap-switch button').forEach(button=>button.addEventListener('click',()=>{state.cap=button.dataset.cap;activate('#cap-switch button','cap',state.cap);renderTail();}));
  $('export-cohort').addEventListener('click',()=>{
    const g=M.aggregate(M.filter(rows,state));
    const result={age:state.age==='all'?'All ages':D.ageLabels[+state.age],region:state.region==='all'?'All regions':D.regions[+state.region],vehicle:state.vehicle==='all'?'All vehicles':D.vehicleLabels[+state.vehicle],model:names[state.model],policies:g.policies,exposure:g.exposure,observed_loss:g.loss,predicted_loss:g[state.model],predicted_actual:M.ratio(g[state.model],g.loss),scope:'Held-out policies; EUR 200000 cap'};
    const csv=Object.keys(result).join(',')+'\n'+Object.values(result).map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')+'\n';
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download='actuarial-cohort-summary.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  renderSurface();renderCohort();renderTail();
})();
