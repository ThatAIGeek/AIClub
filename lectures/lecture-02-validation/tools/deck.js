const PptxGenJS = require('pptxgenjs');

// ── палітра: приладова панель / телеметрія ────────────────────────────────
const C = {
  dark:'101820', darkPanel:'1B2530', white:'FFFFFF', panel:'F1F3F5', panel2:'E7EBEE',
  ink:'14181D', ink2:'3B454F', muted:'6B7581', line:'D3D9DE', lineD:'34404C',
  train:'2F6289', val:'8F6A0C', test:'5B3B70', alarm:'A83620', ok:'2C6349', accent:'A8520C',
  trainT:'DEE9F2', valT:'F3EACF', testT:'E9E1F1', alarmT:'F7E3DD', okT:'DCE9E3',
  accentL:'E09A4E', trainL:'7FB0D8', alarmL:'E4866B', okL:'72BB9A', valL:'D9AE55'
};
const F = { sans:'Arial', mono:'Courier New' };
const W = 13.333, H = 7.5, M = 0.62, CW = W - 2*M;

let pres, n = 0;

function slide(o){
  o = o || {};
  const s = pres.addSlide();
  n += 1;
  s.background = { color: o.dark ? C.dark : C.white };
  if (o.eyebrow) s.addText(o.eyebrow.toUpperCase(), {
    x:M, y:0.3, w:CW-1.2, h:0.26, fontSize:10.5, bold:true, charSpacing:1.6,
    color:o.dark?C.accentL:C.accent, fontFace:F.mono, isTextBox:true, margin:0, valign:'middle'
  });
  if (o.title) s.addText(o.title, {
    x:M, y:0.6, w:CW, h:o.titleH||0.72, fontSize:o.titleSize||28, bold:true,
    color:o.dark?C.white:C.ink, fontFace:F.sans, isTextBox:true, margin:0, valign:'top'
  });
  s.addText(String(n), {
    x:W-1.05, y:0.28, w:0.45, h:0.26, fontSize:10.5, color:o.dark?C.muted:C.muted,
    fontFace:F.mono, align:'right', isTextBox:true, margin:0, valign:'middle'
  });
  if (o.notes) s.addNotes(o.notes);
  return s;
}

function txt(s, str, o){
  s.addText(str, Object.assign({ fontFace:F.sans, isTextBox:true, margin:0, valign:'top' }, o));
}

// рядки з інлайновим форматуванням: ['звичайний', {t:'жирний', b:1}, {t:'код', m:1}]
function runs(arr, base){
  base = base || {};
  return (Array.isArray(arr) ? arr : [arr]).map(function(p){
    if (typeof p === 'string') return { text:p, options:Object.assign({}, base) };
    const o = Object.assign({}, base);
    if (p.b) o.bold = true;
    if (p.i) o.italic = true;
    if (p.m) o.fontFace = F.mono;
    if (p.c) o.color = p.c;
    if (p.sz) o.fontSize = p.sz;
    if (p.br) o.breakLine = true;
    return { text:p.t, options:o };
  });
}

function bullets(s, items, o){
  const base = { fontSize:o.size||12, color:o.color||C.ink2, fontFace:F.sans };
  const para = items.map(function(it, i){
    const r = runs(it, base);
    r[0].options = Object.assign({}, r[0].options, { bullet:{ code:'2022', indent:14 }, paraSpaceAfter:o.gap===undefined?7:o.gap });
    r[r.length-1].options = Object.assign({}, r[r.length-1].options, { breakLine: i < items.length-1 });
    return r;
  }).reduce(function(a,b){ return a.concat(b); }, []);
  s.addText(para, { x:o.x, y:o.y, w:o.w, h:o.h, isTextBox:true, margin:0, valign:'top', lineSpacingMultiple:1.1 });
}

function chip(s, label, o){
  s.addText(label.toUpperCase(), {
    x:o.x, y:o.y, w:o.w||1.5, h:0.26, fontSize:9, bold:true, charSpacing:0.8,
    color:o.color||C.muted, fill:{ color:o.fill||C.panel2 }, fontFace:F.mono,
    align:'center', valign:'middle', isTextBox:true, margin:0, rectRadius:0.04
  });
}

function card(s, o){
  s.addShape('roundRect', {
    x:o.x, y:o.y, w:o.w, h:o.h, rectRadius:0.05,
    fill:{ color:o.tint || C.panel },
    line:{ color:o.border || (o.tint ? o.tint : C.line), width:1 }
  });
  const px = o.x + 0.24, pw = o.w - 0.48;
  let ty = o.y + 0.18;
  if (o.tag){ chip(s, o.tag, { x:px, y:ty, w:o.tagW||1.55, fill:o.tagFill, color:o.tagColor }); ty += 0.38; }
  if (o.stat){
    txt(s, o.stat, { x:px, y:ty, w:pw, h:o.statH||0.9, fontSize:o.statSize||44, bold:true,
      color:o.statColor||C.ink, fontFace:F.mono, align:o.statAlign||'left' });
    ty += (o.statH||0.9) + 0.06;
  }
  if (o.title){
    txt(s, o.title, { x:px, y:ty, w:pw, h:o.titleH||0.3, fontSize:o.titleSize||13.5, bold:true, color:o.titleColor||C.ink });
    ty += (o.titleH||0.3) + 0.1;
  }
  if (o.body){
    s.addText(runs(o.body, { fontSize:o.size||11.5, color:o.bodyColor||C.ink2, fontFace:F.sans }),
      { x:px, y:ty, w:pw, h:Math.max(0.3, o.y + o.h - ty - 0.16), isTextBox:true, margin:0, valign:'top', lineSpacingMultiple:1.12 });
    return;
  }
  if (o.bullets){
    bullets(s, o.bullets, { x:px, y:ty, w:pw, h:Math.max(0.3, o.y + o.h - ty - 0.16), size:o.size||11.5, gap:o.gap });
  }
}

// смуги розбиття — єдина колірна схема курсу
function foldRow(s, o){
  const lw = o.labelW === undefined ? 1.05 : o.labelW;
  if (o.label) txt(s, o.label, { x:o.x, y:o.y, w:lw-0.12, h:o.h, fontSize:o.labelSize||9.5,
    color:C.muted, fontFace:F.mono, align:'right', valign:'middle' });
  const bx = o.x + lw, bw = o.w - lw;
  const total = o.cells.reduce(function(a,c){ return a + (c.span||1); }, 0);
  const gap = 0.045;
  let cx = bx;
  o.cells.forEach(function(c){
    const cw = (bw - gap*(o.cells.length-1)) * (c.span||1)/total;
    s.addShape('roundRect', { x:cx, y:o.y, w:cw, h:o.h, rectRadius:0.03,
      fill:{ color:c.fill||C.train }, line:{ color:c.border||c.fill||C.train, width: c.border?1.5:0.75 } });
    if (c.text) txt(s, c.text, { x:cx, y:o.y, w:cw, h:o.h, fontSize:c.size||9,
      color:c.color||C.white, fontFace:F.mono, align:'center', valign:'middle' });
    cx += cw + gap;
  });
}

function legend(s, o){
  let x = o.x;
  o.items.forEach(function(it){
    s.addShape('roundRect', { x:x, y:o.y+0.055, w:0.26, h:0.12, rectRadius:0.02, fill:{color:it.c}, line:{color:it.c,width:0.5} });
    txt(s, it.t, { x:x+0.34, y:o.y, w:it.w||1.35, h:0.24, fontSize:10, color:o.color||C.muted, fontFace:F.sans, valign:'middle' });
    x += 0.34 + (it.w||1.35) + 0.22;
  });
}

function flowStep(s, o){
  s.addShape('roundRect', { x:o.x, y:o.y, w:o.w, h:o.h||0.42, rectRadius:0.05,
    fill:{ color:o.fill||C.white }, line:{ color:o.border||C.line, width:1 } });
  txt(s, o.t, { x:o.x+0.06, y:o.y, w:o.w-0.12, h:o.h||0.42, fontSize:o.size||10.5,
    color:o.color||C.ink2, fontFace:F.mono, align:'center', valign:'middle' });
}
function arrow(s, x, y, color){
  txt(s, '→', { x:x, y:y, w:0.26, h:0.42, fontSize:13, color:color||C.muted, fontFace:F.sans, align:'center', valign:'middle' });
}

function lead(s, arr, o){
  o = o || {};
  s.addText(runs(arr, { fontSize:o.size||14, color:o.color||C.ink, fontFace:F.sans }),
    { x:o.x||M, y:o.y, w:o.w||CW, h:o.h||0.5, isTextBox:true, margin:0, valign:'middle', lineSpacingMultiple:1.1 });
}

function src(s, str, o){
  txt(s, str, { x:(o&&o.x)||M, y:o.y, w:(o&&o.w)||CW, h:0.26, fontSize:9,
    color:(o&&o.color)||C.muted, fontFace:F.mono, valign:'middle' });
}

const chartBase = {
  chartColors:[C.train], showLegend:false, showTitle:false,
  catAxisLabelColor:C.muted, valAxisLabelColor:C.muted,
  catAxisLabelFontFace:F.mono, valAxisLabelFontFace:F.mono,
  catAxisLabelFontSize:9, valAxisLabelFontSize:9,
  valGridLine:{ color:C.line, size:0.5 }, catGridLine:{ style:'none' },
  axisLineColor:C.line, lineDataSymbol:'none', lineSmooth:false, dataBorder:{pt:0}
};

module.exports = { C, F, W, H, M, CW, slide, txt, runs, bullets, chip, card, foldRow,
  legend, flowStep, arrow, lead, src, chartBase,
  init:function(p){ pres = p; n = 0; }, count:function(){ return n; } };
