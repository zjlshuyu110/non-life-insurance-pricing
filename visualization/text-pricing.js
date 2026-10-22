"use strict";
(() => {
  const data = window.TEXT_PRICING_DATA, $ = id => document.getElementById(id);
  const entries = [...data.cases, ...data.public], m = data.metrics;
  const currency = value => new Intl.NumberFormat("en-IE", {style:"currency", currency:"EUR", maximumFractionDigits:2}).format(value);
  const escape = s => String(s).replace(/[&<>"']/g, x => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[x]));
  const labels = {DrivAge:"Driver age / 驾驶人年龄", VehAge:"Vehicle age / 车龄", BonusMalus:"Bonus–malus", VehGas:"Fuel / 燃料", AnnualKm:"Annual km / 预计年里程", Usage:"Vehicle use / 用途"};
  let selected;
  $("stat-priced").textContent = m.priced;
  $("stat-review").textContent = m.review;
  $("stat-blocked").textContent = m.blocked;
  $("stat-public").textContent = m.public_exact_matches + " / " + m.public_questions;
  $("public-score").textContent = m.public_exact_matches + " / " + m.public_questions;
  $("field-score").textContent = m.correct + " / " + m.fields;
  $("field-detail").textContent = "Validated values and correct abstentions. " + m.present_correct + "/" + m.present + " present fields recovered; " + m.absent_abstained + "/" + m.absent + " absent or unsupported fields left empty.";
  $("fixed-profile").textContent = JSON.stringify(data.config.reference_profile, null, 2);
  for (const [name, items] of [["Authored application demos", data.cases], ["Public synthetic PDFs · extraction only", data.public]]) {
    const group = document.createElement("optgroup"); group.label = name;
    for (const c of items) { const option = document.createElement("option"); option.value = c.id; option.textContent = c.title; group.append(option); }
    $("case-select").append(group);
  }
  function highlight(field, button) {
    for (const b of document.querySelectorAll(".field")) { b.classList.remove("selected"); b.setAttribute("aria-pressed", "false"); }
    if (button) { button.classList.add("selected"); button.setAttribute("aria-pressed", "true"); }
    const out = $("document-text"), text = selected.text;
    out.replaceChildren();
    if (field && field.answer && text.slice(field.start, field.end) === field.answer) {
      out.append(document.createTextNode(text.slice(0, field.start)));
      const mark = document.createElement("mark"); mark.textContent = text.slice(field.start, field.end); out.append(mark);
      out.append(document.createTextNode(text.slice(field.end)));
      out.scrollTop = Math.max(0, mark.offsetTop - out.offsetTop - 100);
    } else { out.textContent = text; out.scrollTop = 0; }
  }
  function fieldButton(key, f, isPublic) {
    const button = document.createElement("button"); button.type = "button"; button.className = "field"; button.setAttribute("aria-pressed", "false");
    const value = isPublic ? (f.answer || "No answer") : (f.value ?? "Unknown / 待确认");
    const status = isPublic ? (f.exact_match ? "raw exact match" : "raw mismatch / no answer") : f.status.replaceAll("_", " ");
    const color = (isPublic ? f.exact_match : f.status === "accepted") ? "ok" : "warn";
    const role = isPublic ? "EXTRACTION ONLY" : f.used_by_glm ? "GLM INPUT" : "RESEARCH ONLY · NO PRICE EFFECT";
    button.innerHTML = '<div class="field-key">' + escape(labels[key] || key) + '</div><div class="field-top"><span class="field-value">' + escape(value) + '</span></div><div class="field-meta"><span class="' + color + '">' + escape(status) + '</span><span>score ' + f.score.toFixed(2) + '</span></div><div class="field-meta"><span>' + role + '</span></div>';
    if (!isPublic && f.value === null && f.answer) { const note = document.createElement("div"); note.className="raw-answer"; note.textContent="Rejected raw answer: “" + f.answer + "”"; button.append(note); }
    if (isPublic) { const note=document.createElement("div"); note.className="raw-answer"; note.textContent="Expected: " + f.expected; button.append(note); }
    button.addEventListener("click", () => highlight(f, button)); return button;
  }
  function render(id) {
    selected = entries.find(c => c.id === id) || entries[0];
    $("case-select").value = selected.id;
    const pub = selected.status === "extraction_only";
    $("extraction-caption").textContent = pub ? "Raw reader answers · compared with source labels" : "Neural answer → schema check → typed value";
    $("pricing-profile").hidden = pub;
    $("doc-badge").textContent = pub ? "PUBLIC · SYNTHETIC" : "AUTHORED · SYNTHETIC";
    $("case-origin").textContent = pub ? "Strata Insurance Corpus · Nikolai Sachok · CC BY 4.0 · unchanged source PDF" : selected.source;
    $("doc-meta").textContent = pub ? "Issued policy document · extraction benchmark only" : "Document: " + selected.document_type + " | Available: " + selected.available_at.slice(0,10) + " | Quote: " + selected.quote_at.slice(0,10);
    $("public-source").hidden = !pub;
    if (pub) $("public-source").href = selected.source_url;
    highlight(null, null);
    $("fields").replaceChildren();
    if (pub) { for (const f of selected.answers) $("fields").append(fieldButton(f.field, f, true)); }
    else { for (const [key,f] of Object.entries(selected.fields)) $("fields").append(fieldButton(key,f,false)); }
    if (!pub && selected.status === "blocked") $("fields").textContent = "Input gate stopped processing before the AI model was called.";
    const out = $("pricing-output");
    if (selected.prediction) {
      const p=selected.prediction, oracle=selected.oracle_prediction;
      out.innerHTML='<span class="label teal">FIELDS PASSED · ILLUSTRATIVE PROFILE</span><p class="price-label">ANNUAL PURE PREMIUM</p><div class="price">'+currency(p.FrequencySeverity)+'<small> / yr</small></div><p class="caption">Poisson frequency × Gamma severity</p><div class="component"><span>Claim frequency / yr</span><b>'+p.Frequency.toFixed(4)+'</b></div><div class="component"><span>Cost per claim</span><b>'+currency(p.Severity)+'</b></div><div class="component"><span>Tweedie GLM</span><b>'+currency(p.Tweedie)+'</b></div>';
      if(oracle){const note=document.createElement("div");note.className="oracle";note.textContent="Same GLM with authored gold fields: "+currency(oracle.FrequencySeverity)+". Difference: "+currency(p.FrequencySeverity-oracle.FrequencySeverity)+".";out.append(note);}
    } else {
      out.replaceChildren();const gate=document.createElement("div");gate.className="gate";
      const title=document.createElement("b");title.textContent=pub?"Extraction only":selected.status==="blocked"?"Blocked before extraction":"Manual review required";gate.append(title);
      for(const reason of (pub?[selected.pricing_reason] : selected.reasons)){const p=document.createElement("p");p.textContent=reason;gate.append(p);}
      out.append(gate);
    }
    const insights={
      commuter:"四个已有定价字段来自文本；其余五个字段来自明确展示的固定示例档案。AI 没有生成价格系数，输出由已拟合的 GLM 计算。",
      high_mileage:"8,000 → 40,000 km: both cases produce "+currency(data.cases.find(c=>c.id==="commuter").prediction.FrequencySeverity)+". 年里程没有进入当前 GLM，因此不会改变价格。这展示模型边界；需要新的关联赔款数据才能学习里程影响。",
      missing_extra:"The raw model calls the vehicle use “diesel” (score ≈ 0.72). Schema validation rejects it. 用途被保留为空；它不属于当前 GLM 所需字段，因此不阻断这次示例计算。",
      delivery:"车辆用途已抽取，但 delivery 没有对应的定价系数。这里的价格变化来自年龄、车龄、燃料和 bonus–malus；不能归因于配送用途。",
      conflict:"Two explicit driver ages conflict. The rule sends this example for review even if the model picks a confident span. 此规则只覆盖有限的明确矛盾表达，不等于完整的语义矛盾检测。",
      claim:"The source is marked as a post-loss claim report. It is blocked from the underwriting path. 材料类型由可信的接收环节提供；不能让 AI 自行把事后材料改成投保材料。",
      late:"This document became available after the quote timestamp. It is blocked to prevent looking into the future.",
      electric:"The fitted French model supports Diesel / Regular. An electric vehicle cannot be silently mapped into one of those categories."
    };
    $("case-insight").textContent=pub?"Public does not mean real: this contract is synthetic. The reader’s raw successes and failures are shown without correction. 它没有连接到 freMTPL2，也不被当作真实投保文本与赔款的配对样本。":insights[selected.id] || (selected.status==="review"?"A required pricing field is unknown or rejected. The GLM handoff stops; no silent median, default bonus or invented answer is used.":"Four validated text fields enter the existing GLM. Mileage and usage remain research-only features. 本例不构成模型预测能力提升的证据。");
    $("raw-json").textContent=JSON.stringify(selected,null,2);
  }
  $("case-select").addEventListener("change",e=>{render(e.target.value);history.replaceState(null,"","#"+selected.id);});
  $("mileage-pair").addEventListener("click",()=>{render("high_mileage");history.replaceState(null,"","#high_mileage");});
  $("download-json").addEventListener("click",()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(selected,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=selected.id+".json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
  window.addEventListener("hashchange",()=>render(location.hash.slice(1)));
  render(location.hash.slice(1));
})();
