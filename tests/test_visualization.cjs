const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const context={window:{}};
vm.runInNewContext(fs.readFileSync(path.join(root,'docs/assets/study-data.js'),'utf8'),context);
const data=context.window.STUDY_DATA;
const M=require('../visualization/engine.js');
const rows=M.decode(data);
const all={age:'all',region:'all',vehicle:'all'};
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-5*Math.max(1,Math.abs(b)),`${a} != ${b}`);
test('browser aggregates reproduce the actual test totals and model deviances',()=>{
  const g=M.aggregate(rows);
  assert.equal(g.policies,169504);
  for(const [model,key] of [['FrequencySeverity','fs'],['Tweedie','tw'],['Baseline','base']]){
    const r=data.globalMetrics.find(r=>r.model===model);
    near(g.loss,r.actual_loss);near(g[key],r.predicted_loss);near(g['dev_'+key]/g.exposure,r['deviance_p1.9']);
  }
});
test('age, region and vehicle partitions conserve policy counts and loss',()=>{
  const total=M.aggregate(rows);
  for(const [key,count] of [['age',7],['region',22],['vehicle',6]]){
    const groups=M.group(rows,key,count);
    assert.equal(groups.reduce((s,g)=>s+g.policies,0),total.policies);
    near(groups.reduce((s,g)=>s+g.loss,0),total.loss);
  }
});
test('linked filtering intersects dimensions and reset recovers the full view',()=>{
  const filtered=M.filter(rows,{age:'0',region:'0',vehicle:'2'});
  assert.ok(filtered.length>0);
  assert.ok(filtered.every(r=>r.age===0&&r.region===0&&r.vehicle===2));
  assert.equal(M.aggregate(M.filter(rows,all)).policies,169504);
  assert.equal(M.aggregate([]).exposure,0);
  assert.equal(M.ratio(4,0),null);
});
test('every loss-cap selector resolves to exactly three real model results',()=>{
  for(const cap of ['50000','200000','Uncapped']){
    const scenarios=data.sensitivity.filter(r=>r.loss_cap===cap);
    assert.equal(scenarios.length,3);
    for(const r of scenarios)assert.ok(r.predicted_actual>0&&Number.isFinite(r.predicted_actual));
  }
});
test('both model surfaces have 777 finite positive predictions and valid slider coordinates',()=>{
  const s=data.surface;
  assert.equal(s.ages.length*s.bonuses.length,777);
  assert.ok(s.ages.includes(40)&&s.bonuses.includes(50));
  for(const key of ['fs','tw']){
    assert.equal(s.values[key].length,s.ages.length);
    for(const row of s.values[key]){assert.equal(row.length,s.bonuses.length);assert.ok(row.every(v=>v>0&&Number.isFinite(v)));}
  }
});
