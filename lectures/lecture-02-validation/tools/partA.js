const D = require('./deck');
const { C, F, W, M, CW, slide, txt, card, foldRow, legend, flowStep, arrow, lead, src, chip, bullets, runs } = D;

module.exports = function(pres){
  const C3 = (CW - 0.56)/3, C2 = (CW - 0.32)/2, C4 = (CW - 0.72)/4;
  let s;

  // 1 ── титул
  s = slide({ dark:true, eyebrow:'Лекція 2 · курс машинного навчання',
    notes:'Хронометраж 80 хв: 0 хук 6 · 1 навіщо розбивати 8 · 2 схеми CV 14 · 3 витік 20 · 4 дисбаланс 14 · 5 калібрування 10 · 6 офлайн vs ціль 5 · 7 чек-ліст 3. Якщо пара 2×40 — перерва після блоку 3.' });
  txt(s, 'Валідація моделей\nта методологія експерименту', { x:M, y:1.15, w:CW, h:1.7, fontSize:40, bold:true, color:C.white, lineSpacingMultiple:1.05 });
  s.addText(runs(['Лекція 1 відповіла, ', {t:'чим міряти',b:1,c:C.accentL}, '. Ця відповідає, ',
    {t:'як організувати вимірювання, щоб число означало те, що ми думаємо',b:1,c:C.white},
    '. Лекція 3 відповість, звідки помилка береться.'], { fontSize:14, color:'C7CFD8', fontFace:F.sans }),
    { x:M, y:3.0, w:8.6, h:1.0, isTextBox:true, margin:0, valign:'top', lineSpacingMultiple:1.2 });
  const t1y = 4.35, t1h = 2.3;
  card(s, { x:M, y:t1y, w:C3, h:t1h, tint:C.darkPanel, border:C.lineD, titleColor:C.white, bodyColor:'AEB9C4',
    title:'Наскрізна рамка', body:[{t:'«The first principle is that you must not fool yourself — and you are the easiest person to fool.»',i:1,br:1},{t:' ',br:1},{t:'R. Feynman, Caltech, 1974',m:1,c:C.accentL}] });
  card(s, { x:M+C3+0.28, y:t1y, w:C3, h:t1h, tint:C.darkPanel, border:C.lineD, titleColor:C.white,
    title:'Чотири способи обдурити себе', bullets:[
      'розбити дані не за тією одиницею', 'пустити інформацію з майбутнього в минуле',
      'міряти метрикою, не чутливою до перекосу', 'повірити в некалібрований score'], size:11, gap:4 });
  card(s, { x:M+2*(C3+0.28), y:t1y, w:C3, h:t1h, tint:C.darkPanel, border:C.lineD, titleColor:C.white, bodyColor:'AEB9C4',
    title:'Результати навчання', body:[{t:'РН5 · РН6 · ПРН3',m:1,c:C.accentL,br:1},{t:' ',br:1},'Перевіряються на захисті ЛР2 та на екзамені.'] });

  // 2 ── два числа
  s = slide({ dark:true, eyebrow:'Блок 0 · хук · 6 хв', title:'Два числа',
    notes:'Руки, не оцінюється. Дати 30–40 секунд на суперечку. Правильна відповідь — методологія. Числа наведені як типова величина ефекту; задокументовані аналоги — на слайді 4.' });
  card(s, { x:M, y:1.75, w:C2, h:2.6, tint:C.darkPanel, border:C.lineD,
    tag:'випадкове розбиття', tagFill:'22323F', tagColor:C.trainL, tagW:2.6,
    stat:'0.94', statColor:C.trainL, statSize:72, statH:1.35, statAlign:'center',
    body:[{t:'mAP',m:1,c:C.muted}], size:12 });
  card(s, { x:M+C2+0.32, y:1.75, w:C2, h:2.6, tint:C.darkPanel, border:C.lineD,
    tag:'розбиття за місіями', tagFill:'2E2436', tagColor:'B693D4', tagW:2.6,
    stat:'0.41', statColor:'B693D4', statSize:72, statH:1.35, statAlign:'center',
    body:[{t:'mAP',m:1,c:C.muted}], size:12 });
  lead(s, [{t:'Одна модель. Один код. Один датасет.',b:1,c:C.white}, ' Різниця тільки в тому, як розбили дані.'], { y:4.7, size:17 });
  lead(s, ['Питання в зал: хто винен — модель, дані чи методологія?'], { y:5.5, size:15, color:C.accentL });

  // 3 ── два сусідні кадри
  s = slide({ eyebrow:'Блок 0 · хук · 6 хв', title:'Чи це два незалежні приклади?',
    notes:'Звідси рамка лекції: п\'ять тем програми подаються не як окремі техніки, а як способи самообману і захист від них.' });
  const fw = 3.9, fh = 2.15, fy = 1.72, fxA = 2.05, fxB = fxA + fw + 1.45;
  [fxA, fxB].forEach(function(fx, i){
    s.addShape('roundRect', { x:fx, y:fy, w:fw, h:fh, rectRadius:0.04, fill:{color:C.panel}, line:{color:C.line, width:1} });
    s.addShape('line', { x:fx+0.2, y:fy+1.5, w:fw-0.4, h:0, line:{color:C.line, width:1} });
    s.addShape('line', { x:fx+0.45, y:fy+1.15, w:1.1, h:-0.28, line:{color:C.muted, width:1.25} });
    s.addShape('line', { x:fx+1.55, y:fy+0.87, w:1.0, h:0.22, line:{color:C.muted, width:1.25} });
    s.addShape('line', { x:fx+2.55, y:fy+1.09, w:0.9, h:-0.18, line:{color:C.muted, width:1.25} });
    s.addShape('rect', { x:fx+1.72+(i?0.06:0), y:fy+0.98, w:0.42, h:0.26, fill:{color:C.test}, line:{color:C.test,width:0.5} });
    s.addShape('rect', { x:fx+1.58+(i?0.06:0), y:fy+0.84, w:0.72, h:0.56, fill:{type:'none'}, line:{color:C.alarm, width:1.5} });
    txt(s, i ? '00:04:12.067' : '00:04:12.033', { x:fx, y:fy+fh+0.06, w:fw, h:0.28, fontSize:11, color:C.muted, fontFace:F.mono, align:'center' });
  });
  s.addShape('line', { x:fxA+fw+0.22, y:fy+1.0, w:1.0, h:0, line:{color:C.accent, width:2, endArrowType:'triangle'} });
  txt(s, '33 мс', { x:fxA+fw+0.12, y:fy+0.55, w:1.2, h:0.3, fontSize:12.5, bold:true, color:C.accent, fontFace:F.mono, align:'center' });
  chip(s, 'train', { x:fxA+fw/2-0.8, y:fy+fh+0.42, w:1.6, fill:C.trainT, color:C.train });
  chip(s, 'test',  { x:fxB+fw/2-0.8, y:fy+fh+0.42, w:1.6, fill:C.testT, color:C.test });
  txt(s, 'одне відео, 30 fps, той самий борт, та сама ділянка місцевості', { x:M, y:4.72, w:CW, h:0.3, fontSize:11.5, color:C.alarm, fontFace:F.sans, align:'center' });
  lead(s, ['Якщо один із цих кадрів у train, а другий у test — ви виміряли ', {t:'не узагальнення, а пам\'ять',b:1,c:C.alarm}, '.'], { y:5.35, size:16 });
  lead(s, ['Обіцянка лекції: після неї ви дивитеся на чужий результат і питаєте «а як ви розбивали?» раніше, ніж «а яка у вас архітектура?».'], { y:6.05, size:13, color:C.ink2 });

  // 4 ── це не гіпотетичні числа
  s = slide({ eyebrow:'Блок 0 · хук · 6 хв', title:'Це не гіпотетичні числа',
    notes:'Оборонній аудиторії — перші два рядки. Тим, хто робить LLM-агентів, — третій: та сама помилка, одиниця незалежності — репозиторій, а «витік» називається контамінацією бенчмарку.' });
  const hd = { bold:true, color:C.muted, fontSize:10.5, fontFace:F.sans, fill:{color:C.white} };
  const cell = { fontSize:11.5, color:C.ink2, fontFace:F.sans, valign:'middle' };
  const mono = Object.assign({}, cell, { fontFace:F.mono, fontSize:11 });
  s.addTable([
    [{text:'Задача',options:hd},{text:'«Наївне» розбиття',options:hd},{text:'Чесне розбиття',options:hd},{text:'Одиниця незалежності',options:hd}],
    [{text:'Тип дрона за радіосигналом, DroneRF',options:cell},{text:'macro-F1 0.74',options:Object.assign({},mono,{color:C.train})},{text:'0.46 — рівень випадковості',options:Object.assign({},mono,{color:C.test})},{text:'запис',options:cell}],
    [{text:'Акустичне виявлення дрона, EchoHawk',options:cell},{text:'Pd 0.937 @ FAR 5 %',options:Object.assign({},mono,{color:C.train})},{text:'0.875',options:Object.assign({},mono,{color:C.test})},{text:'сесія запису',options:cell}],
    [{text:'Пошук файлу з багом за текстом issue',options:cell},{text:'76 % на SWE-bench Verified',options:Object.assign({},mono,{color:C.train})},{text:'53 % на інших репозиторіях',options:Object.assign({},mono,{color:C.test})},{text:'репозиторій',options:cell}],
    [{text:'Класифікація Android-малварі',options:cell},{text:'F1 до 0.99',options:Object.assign({},mono,{color:C.train})},{text:'суттєво нижче за часового розбиття',options:Object.assign({},mono,{color:C.test})},{text:'час появи зразка',options:cell}]
  ], { x:M, y:1.65, w:CW, colW:[3.9,2.85,3.4,1.94], rowH:0.62, border:{type:'solid',color:C.line,pt:0.5},
       fill:{color:C.white}, margin:[6,10,6,10], valign:'middle' });
  lead(s, ['У всіх чотирьох випадках падіння дає не інша модель, а інша ', {t:'процедура вимірювання',b:1}, '.'], { y:4.95, size:14 });
  src(s, 'DroneRF: arXiv 2607.01025 · EchoHawk: arXiv 2606.29589 · SWE-Bench Illusion: arXiv 2506.12286 · TESSERACT: USENIX Security \'19', { y:5.6 });

  // 5 ── емпіричний vs істинний ризик
  s = slide({ eyebrow:'Блок 1 · навіщо розбивати · 8 хв', title:'Емпіричний ризик vs істинний ризик',
    notes:'Достатньо: train error падає монотонно, test error має мінімум, розрив між ними — це те, що ми хочемо виміряти, а не сховати. Лекція 3 розкладе цей розрив на зсув і дисперсію.' });
  s.addChart(pres.ChartType.line, [
    { name:'помилка на train', labels:['','','','','','','','','',''], values:[0.62,0.48,0.38,0.30,0.24,0.19,0.15,0.12,0.09,0.07] },
    { name:'помилка на test',  labels:['','','','','','','','','',''], values:[0.66,0.55,0.47,0.42,0.40,0.41,0.45,0.52,0.62,0.74] }
  ], Object.assign({}, D.chartBase, { x:M, y:1.6, w:7.0, h:4.05, chartColors:[C.train, C.test],
      showLegend:true, legendPos:'b', legendColor:C.muted, legendFontFace:F.sans, legendFontSize:10,
      lineSize:2.5, valAxisMaxVal:0.8, valAxisMinVal:0, valAxisTitle:'помилка', showValAxisTitle:true,
      valAxisTitleColor:C.muted, valAxisTitleFontSize:10, valAxisTitleFontFace:F.sans,
      catAxisTitle:'ємність моделі / кількість підганянь', showCatAxisTitle:true,
      catAxisTitleColor:C.muted, catAxisTitleFontSize:10, catAxisTitleFontFace:F.sans }));
  card(s, { x:M+7.25, y:1.6, w:CW-7.25, h:1.95, tint:C.panel, title:'Головне твердження',
    body:['Помилка на train — ', {t:'зміщена, оптимістична',b:1}, ' оцінка істинного ризику. Зміщення росте з ємністю моделі й з кількістю разів, коли ви щось підганяли.'] });
  card(s, { x:M+7.25, y:3.72, w:CW-7.25, h:1.93, tint:C.white, border:C.line,
    body:['Ми ніколи не бачимо істинний ризик. Ми бачимо його ', {t:'оцінку',i:1}, ' на скінченній вибірці. Уся методологія — про те, щоб ця оцінка була незміщеною і щоб ми знали її дисперсію.'] });

  // 6 ── три ролі вибірок
  s = slide({ eyebrow:'Блок 1 · навіщо розбивати · 8 хв', title:'Три ролі вибірок',
    notes:'Ця колірна схема (train синій / val вохра / test сливовий) повертається в лекціях 4, 13 і 14. Попросити студентів запам\'ятати саме її.' });
  foldRow(s, { x:M, y:1.62, w:CW, h:0.62, labelW:0, cells:[
    { span:6, fill:C.train, text:'TRAIN · 60 %', size:11 },
    { span:2, fill:C.val,   text:'VAL · 20 %',   size:11 },
    { span:2, fill:C.test,  text:'TEST · 20 %',  size:11 } ] });
  const r6 = [
    { tag:'train', tf:C.trainT, tc:C.train, t:'Налаштовує параметри', b:['Ваги, коефіцієнти, розбиття дерева. Дивимося скільки завгодно разів.'] },
    { tag:'validation', tf:C.valT, tc:C.val, t:'Обирає гіперпараметри й модель', b:['Глибину дерева, C, кількість компонент, поріг. Дивимося багато разів — і саме тому вона теж стає оптимістичною.'] },
    { tag:'test', tf:C.testT, tc:C.test, t:'Дає ОДНУ незміщену оцінку', b:['Дивимося один раз. Після рішення. Перед звітом.'] }
  ];
  r6.forEach(function(c,i){ card(s, { x:M+i*(C3+0.28), y:2.55, w:C3, h:2.35, tint:C.white, border:C.line,
    tag:c.tag, tagFill:c.tf, tagColor:c.tc, tagW:1.9, title:c.t, body:c.b }); });
  lead(s, ['Найчастіша помилка студентських робіт — ', {t:'злити val і test в одну вибірку',b:1,c:C.alarm}, ' і звітувати число, за яким же й обирали модель.'], { y:5.35, size:14.5 });

  // 7 ── одноразовий патрон
  s = slide({ eyebrow:'Блок 1 · навіщо розбивати · 8 хв', title:'Тестова вибірка — це одноразовий патрон',
    notes:'Оцінка максимуму m незалежних гаусівських величин. Не доводити — показати порядок. Це кількісна відповідь на «ну і що, я ж просто подивився».' });
  lead(s, ['Кожен погляд на тест витрачає його. Після третього «а давайте ще перевіримо на тесті» у вас ', {t:'друга валідаційна вибірка',b:1}, ', а не тест.'], { y:1.6, size:14.5, h:0.6 });
  card(s, { x:M, y:2.35, w:C2, h:2.35, tint:C.alarmT, border:C.alarm, title:'Як це виглядає', bullets:[
    '«Спробували 30 конфігурацій, взяли найкращу за тестом.»',
    '«Модель показала 0.78, ми трохи змінили аугментації — стало 0.81.»',
    '«Поріг підібрали так, щоб на тесті вийшло красиво.»'] });
  card(s, { x:M+C2+0.32, y:2.35, w:C2, h:2.35, tint:C.okT, border:C.ok, title:'Як має бути', bullets:[
    'Тест лежить в окремому файлі й не імпортується в ноутбук розробки.',
    'Кількість звернень до тесту фіксується в журналі експериментів.',
    'Витратили тест — збирайте новий або чесно звітуйте, скільки разів дивилися.'] });
  card(s, { x:M, y:4.95, w:CW, h:1.25, tint:C.panel,
    body:['Формально: обираючи найкраще з ', {t:'m',m:1}, ' конфігурацій за тією самою вибіркою, ви зміщуєте оцінку вгору приблизно на ', {t:'σ · √(2 ln m)',m:1,b:1}, '. Для 30 конфігурацій і σ = 2 п.п. це вже близько ', {t:'5 п.п. «безкоштовного» приросту',b:1,c:C.alarm}, '.'], size:13 });

  // 8 ── шум оцінки
  s = slide({ eyebrow:'Блок 1 · ПРН3 · 8 хв', title:'Шум оцінки: різниця 0.91 vs 0.89 — це не різниця',
    notes:'Щоб надійно побачити різницю 2 п.п., потрібно близько 2 000 прикладів у тесті. У більшості студентських ЛР тест — 200–400 прикладів. Отже, всі їхні «покращення» на 1–2 п.п. — шум. Сказати це прямо.' });
  s.addChart(pres.ChartType.line, [
    { name:'півширина 95 % ДІ', labels:['100','200','500','1K','2K','5K','10K'], values:[5.88,4.16,2.63,1.86,1.31,0.83,0.59] }
  ], Object.assign({}, D.chartBase, { x:M, y:1.6, w:7.0, h:4.05, chartColors:[C.train], lineSize:2.5,
      lineDataSymbol:'circle', lineDataSymbolSize:7, showValue:true, dataLabelColor:C.ink2,
      dataLabelFontFace:F.mono, dataLabelFontSize:10, dataLabelPosition:'t', dataLabelFormatCode:'0.0',
      valAxisTitle:'півширина 95 % ДІ, п.п.', showValAxisTitle:true, valAxisTitleColor:C.muted,
      valAxisTitleFontSize:10, valAxisTitleFontFace:F.sans, valAxisMinVal:0,
      catAxisTitle:'розмір тестової вибірки n', showCatAxisTitle:true, catAxisTitleColor:C.muted,
      catAxisTitleFontSize:10, catAxisTitleFontFace:F.sans }));
  card(s, { x:M+7.25, y:1.6, w:CW-7.25, h:2.1, tint:C.panel, title:'На серветці',
    body:[{t:'1,96 · √(0,9 · 0,1 / 200) ≈ 0,042',m:1,b:1,c:C.ink,br:1},{t:' ',br:1},
      'Точність на тесті — біноміальна випадкова величина. При n = 200 і accuracy 0.90 півширина 95 % ДІ ≈ ', {t:'4,2 п.п.',b:1,c:C.alarm}] });
  card(s, { x:M+7.25, y:3.87, w:CW-7.25, h:1.78, tint:C.okT, border:C.ok, title:'Вимога до ЛР',
    body:['Кожна метрика в звіті — з довірчим інтервалом (біноміальний або бутстреп). Порівняння двох моделей без ДІ на захисті не приймається.'] });

  // 9 ── одиниця незалежності
  s = slide({ eyebrow:'Блок 2 · схеми крос-валідації · 14 хв', title:'Одне питання, з якого випливають усі схеми',
    notes:'Не читати таблицю вголос. Показати й одразу перейти до розбору чотирьох рядків: k-fold, stratified, group, time-series. Nested — наприкінці блоку.' });
  txt(s, 'Що є одиницею незалежності у ваших даних?', { x:M, y:1.52, w:CW, h:0.6, fontSize:22, bold:true, color:C.accent });
  const hd9 = { bold:true, color:C.muted, fontSize:10.5, fontFace:F.sans, fill:{color:C.white} };
  const c9  = { fontSize:11.5, color:C.ink2, fontFace:F.sans, valign:'middle' };
  const c9b = Object.assign({}, c9, { bold:true, color:C.ink, fill:{color:C.trainT} });
  const c9t = Object.assign({}, c9, { fill:{color:C.trainT} });
  s.addTable([
    [{text:'Схема',options:hd9},{text:'Коли',options:hd9},{text:'Одиниця незалежності',options:hd9}],
    [{text:'Hold-out',options:c9},{text:'багато даних, швидкий цикл',options:c9},{text:'приклад',options:c9}],
    [{text:'k-fold (k = 5 / 10)',options:c9},{text:'стандарт, менша дисперсія оцінки',options:c9},{text:'приклад',options:c9}],
    [{text:'Stratified k-fold',options:c9},{text:'рідкісні класи, малі вибірки',options:c9},{text:'приклад зі збереженням пропорцій',options:c9}],
    [{text:'Group k-fold / LOGO',options:c9b},{text:'кадри з польоту, об\'єкти, оператори, борти',options:c9t},{text:'місія, борт, об\'єкт, оператор',options:c9b}],
    [{text:'Time-series split (rolling origin)',options:c9},{text:'телеметрія, дрейф, малварь',options:c9},{text:'час',options:c9}],
    [{text:'Nested CV',options:c9},{text:'одночасно підбираємо і звітуємо',options:c9},{text:'два рівні',options:c9}]
  ], { x:M, y:2.3, w:CW, colW:[3.5,4.9,3.69], rowH:0.5, border:{type:'solid',color:C.line,pt:0.5},
       fill:{color:C.white}, margin:[5,10,5,10], valign:'middle' });
  lead(s, ['Схема валідації — це ', {t:'не питання смаку і не питання бібліотеки',b:1}, '. Це відповідь на питання, що саме корелює між вашими прикладами.'], { y:6.1, size:13.5 });

  // 10 ── hold-out і k-fold
  s = slide({ eyebrow:'Блок 2 · схеми крос-валідації · 14 хв', title:'Hold-out і k-fold',
    notes:'Підкреслити: σ по фолдах — це НЕ довірчий інтервал для узагальнення, бо фолди перетинаються по train. Це радше індикатор стабільності. Для ДІ — бутстреп на тесті.' });
  legend(s, { x:M, y:1.52, items:[{c:C.train,t:'train',w:0.75},{c:C.val,t:'validation',w:1.25},{c:C.test,t:'test',w:0.6}] });
  s.addShape('roundRect', { x:M, y:1.95, w:C2, h:1.5, rectRadius:0.05, fill:{color:C.white}, line:{color:C.line,width:1} });
  txt(s, 'Hold-out', { x:M+0.24, y:2.1, w:C2-0.48, h:0.3, fontSize:13.5, bold:true, color:C.ink });
  foldRow(s, { x:M+0.24, y:2.5, w:C2-0.48, h:0.42, labelW:0.95, label:'один поділ',
    cells:[{span:4, fill:C.train, text:'train'},{span:1, fill:C.val, text:'val'}] });
  txt(s, 'Дешево і швидко, але оцінка залежить від того, які саме приклади випали у val.', { x:M+0.24, y:2.98, w:C2-0.48, h:0.5, fontSize:11, color:C.ink2 });
  s.addShape('roundRect', { x:M+C2+0.32, y:1.95, w:C2, h:3.75, rectRadius:0.05, fill:{color:C.white}, line:{color:C.line,width:1} });
  txt(s, 'k-fold, k = 5', { x:M+C2+0.56, y:2.1, w:C2-0.48, h:0.3, fontSize:13.5, bold:true, color:C.ink });
  for (let i=0;i<5;i++){
    const cells = [0,1,2,3,4].map(function(j){ return j===i ? {span:1, fill:C.val, text:'val'} : {span:1, fill:C.train}; });
    foldRow(s, { x:M+C2+0.56, y:2.55+i*0.5, w:C2-0.48, h:0.38, labelW:0.85, label:'фолд '+(i+1), cells:cells });
  }
  txt(s, 'Кожен приклад рівно раз побував у val. Звітуємо середнє і стандартне відхилення по фолдах.', { x:M+C2+0.56, y:5.1, w:C2-0.48, h:0.5, fontSize:11, color:C.ink2 });
  card(s, { x:M, y:3.62, w:C2, h:2.08, tint:C.panel, title:'Правило вибору k',
    body:['Більший k — менше зміщення, але дорожче і ', {t:'сильніша кореляція між фолдами',b:1}, ', тому σ по фолдах недооцінює справжню дисперсію. k = 5 або 10 — компроміс, який працює. LOOCV майже завжди надлишковий.'] });

  // 11 ── stratified
  s = slide({ eyebrow:'Блок 2 · схеми крос-валідації · 14 хв', title:'Stratified k-fold потрібен не «для краси»',
    notes:'Природний місток до блоку 3: стратифікація — інструмент проти дисперсії, а не проти витоку. Студенти регулярно плутають ці дві задачі.' });
  card(s, { x:M, y:1.62, w:7.0, h:1.85, tint:C.alarmT, border:C.alarm, title:'Звичайний k-fold при 2 % позитивів',
    body:['1 000 кадрів, 20 із ціллю. У 5-fold на фолд припадає в середньому 4 позитиви — але випадково легко отримати фолд із ', {t:'двома',b:1}, '. Recall на такому фолді може бути 0.5 або 1.0, і це нічого не означає.'] });
  card(s, { x:M, y:3.65, w:7.0, h:1.5, tint:C.okT, border:C.ok, title:'Stratified k-fold',
    body:['Пропорція класів однакова в кожному фолді. Дисперсія оцінки падає, метрика стає інтерпретовною.'] });
  card(s, { x:M+7.25, y:1.62, w:CW-7.25, h:3.53, tint:C.white, border:C.line, title:'Межа застосовності',
    body:['Стратифікація ', {t:'не рятує від групового витоку',b:1,c:C.alarm}, '. Якщо ті самі 20 позитивів — це 20 кадрів одного об\'єкта з одного прольоту, stratified k-fold сумлінно розкладе їх по всіх фолдах і тим ', {t:'гарантує',i:1}, ' витік.',
      {t:' ',br:1},{t:' ',br:1},
      {t:'StratifiedGroupKFold(...)',m:1,b:1,c:C.accent,br:1},{t:' ',br:1},
      'У scikit-learn є комбінована схема. Саме її треба брати, коли в даних одночасно рідкісний клас і групи.'] });

  // 12 ── group k-fold
  s = slide({ eyebrow:'Блок 2 · схеми крос-валідації · 14 хв', title:'Group k-fold: центральна схема для вашої предметної області',
    notes:'Запитати зал: назвіть свою групу для своїх даних. Якщо студент не може назвати одиницю незалежності — він ще не розуміє задачу, а не методологію.' });
  s.addShape('roundRect', { x:M, y:1.6, w:C2, h:2.85, rectRadius:0.05, fill:{color:C.alarmT}, line:{color:C.alarm,width:1} });
  txt(s, 'Випадкове розбиття кадрів', { x:M+0.24, y:1.76, w:C2-0.48, h:0.3, fontSize:13.5, bold:true, color:C.ink });
  ['місія 1','місія 2','місія 3'].forEach(function(lb,i){
    const pat = [[1,1,0,1,0,1],[0,1,1,0,1,1],[1,0,1,1,0,1]][i];
    foldRow(s, { x:M+0.24, y:2.2+i*0.5, w:C2-0.48, h:0.38, labelW:0.95, label:lb,
      cells: pat.map(function(v){ return v ? {span:1, fill:C.train, border:C.alarm} : {span:1, fill:C.val}; }) });
  });
  txt(s, 'Кожна місія по обидва боки межі. Модель бачила цей борт, це освітлення, цю ділянку і цей шум сенсора.', { x:M+0.24, y:3.78, w:C2-0.48, h:0.5, fontSize:11, color:C.ink2 });
  s.addShape('roundRect', { x:M+C2+0.32, y:1.6, w:C2, h:2.85, rectRadius:0.05, fill:{color:C.okT}, line:{color:C.ok,width:1} });
  txt(s, 'Розбиття за місіями (GroupKFold)', { x:M+C2+0.56, y:1.76, w:C2-0.48, h:0.3, fontSize:13.5, bold:true, color:C.ink });
  ['місія 1','місія 2','місія 3'].forEach(function(lb,i){
    foldRow(s, { x:M+C2+0.56, y:2.2+i*0.5, w:C2-0.48, h:0.38, labelW:0.95, label:lb,
      cells: [0,1,2,3,4,5].map(function(){ return {span:1, fill: i===1 ? C.val : C.train}; }) });
  });
  txt(s, 'Ціла місія цілком по один бік межі. Саме це відповідає питанню «як модель спрацює на наступному вильоті».', { x:M+C2+0.56, y:3.78, w:C2-0.48, h:0.5, fontSize:11, color:C.ink2 });
  lead(s, ['Групу визначає не файл, а ', {t:'джерело кореляції',b:1,c:C.accent}, ': один політ, один борт, одна ділянка місцевості, один день, один оператор, один тепловізор.'], { y:4.8, size:14.5 });
  src(s, 'Стандартна формула програми: розбиття польотних журналів за місіями, а не за кадрами.', { y:5.55 });

  // 13 ── груповий витік у реальних роботах
  s = slide({ eyebrow:'Блок 2 · схеми крос-валідації · 14 хв', title:'Груповий витік у реальних роботах',
    notes:'Три різні сенсори — РЧ, акустика, відео — і той самий механізм помилки. Це аргумент, чому питання «яка одиниця незалежності» універсальніше за будь-яку конкретну техніку.' });
  const r13 = [
    { tag:'радіочастотне виявлення', t:'macro-F1  0.74 → 0.46',
      b:['Неперервні записи нарізали на сегменти й перемішали. Після leave-one-recording-out на DroneRF розпізнавання типу дрона падає до рівня випадкового вгадування з двох класів. Абляція: майже все «покращення» давав сегментний витік.'], s:'arXiv 2607.01025' },
    { tag:'акустичне виявлення', t:'Pd@FAR 5 %  0.937 → 0.875',
      b:['Той самий механізм на попередньо нарізаному публічному датасеті. При FAR 1 % — 0.796 → 0.745. Ефект менший, ніж у РЧ, але систематичний і вимірюваний.'], s:'arXiv 2606.29589' },
    { tag:'аеровідеоспостереження', t:'98,5 % near-duplicate пар',
      b:['У VisDrone сусідні кадри одного прольоту — майже дублікати. Структурно-обізнане розбиття прибирає таку частку перетину між train і val і робить валідацію справді out-of-distribution.'], s:'arXiv 2607.02055' }
  ];
  r13.forEach(function(c,i){
    const x = M+i*(C3+0.28);
    card(s, { x:x, y:1.62, w:C3, h:3.52, tint:C.white, border:C.line,
      tag:c.tag, tagFill:C.alarmT, tagColor:C.alarm, tagW:C3-0.48,
      title:c.t, titleSize:14, titleColor:C.alarm, titleH:0.5, body:c.b, size:11 });
    src(s, c.s, { x:x+0.24, y:5.26, w:C3-0.48 });
  });

  // 14 ── rolling origin
  s = slide({ eyebrow:'Блок 2 · схеми крос-валідації · 14 хв', title:'Часове розбиття: rolling origin',
    notes:'Ковзні статистики й target encoding треба рахувати тільки з даних лівіше межі. Найчастіша реалізація помилки — rolling mean, порахований по всьому ряду до розбиття.' });
  legend(s, { x:M, y:1.5, items:[{c:C.train,t:'train',w:0.75},{c:C.val,t:'validation',w:1.25},{c:C.panel2,t:'ще не сталося',w:1.6}] });
  for (let i=0;i<4;i++){
    const cells = [0,1,2,3,4,5,6].map(function(j){
      if (j < i+2) return { span:1, fill:C.train };
      if (j === i+2) return { span:1, fill:C.val, text:'val' };
      return { span:1, fill:C.panel2 };
    });
    foldRow(s, { x:M, y:1.95+i*0.5, w:7.0, h:0.38, labelW:0.95, label:'крок '+(i+1), cells:cells });
  }
  foldRow(s, { x:M, y:4.0, w:7.0, h:0.34, labelW:0.95, label:'час →',
    cells:['т1','т2','т3','т4','т5','т6','т7'].map(function(t){ return {span:1, fill:C.panel2, text:t, color:C.muted, size:8.5}; }) });
  card(s, { x:M+7.25, y:1.85, w:CW-7.25, h:1.75, tint:C.panel, title:'Коли обов\'язково',
    body:['Телеметрія, прогноз відмов, виявлення малварі, будь-яка задача з дрейфом. Валідація має відтворювати експлуатацію: ', {t:'навчаємось на минулому, працюємо на майбутньому',b:1}, '.'] });
  card(s, { x:M+7.25, y:3.75, w:CW-7.25, h:2.15, tint:C.alarmT, border:C.alarm, title:'Ціна ігнорування',
    body:['У класифікації Android-малварі опубліковані F1 доходили до 0.99. Після усунення часового зміщення (навчання лише на раніших зразках) і просторового (реалістична пропорція малварі) результати виявились суттєво завищеними.',
      {t:' ',br:1},{t:'TESSERACT, USENIX Security \'19 · 129K застосунків за 3 роки',m:1,c:C.muted,sz:9}] });

  // 15 ── nested CV
  s = slide({ eyebrow:'Блок 2 · схеми крос-валідації · 14 хв', title:'Nested CV: коли ви одночасно підбираєте і звітуєте',
    notes:'Вартість: k_зовн × k_внутр навчань. Для 5×5 це 25 моделей. На практиці — або окремий заморожений тест, або nested CV на малих даних, але не «5-fold і звіт по ньому ж».' });
  s.addShape('roundRect', { x:M, y:1.62, w:7.0, h:3.4, rectRadius:0.05, fill:{color:C.white}, line:{color:C.line,width:1} });
  foldRow(s, { x:M+0.24, y:1.85, w:6.52, h:0.42, labelW:1.15, label:'зовнішній',
    cells:[{span:1, fill:C.test, text:'оцінка'},{span:4, fill:C.train, text:'внутрішній цикл'}] });
  for (let i=0;i<3;i++){
    const cells = [0,1,2,3].map(function(j){ return j===i ? {span:1, fill:C.val, text:'val'} : {span:1, fill:C.train}; });
    foldRow(s, { x:M+0.24+1.16, y:2.42+i*0.5, w:6.52-1.16, h:0.38, labelW:1.15, label:'внутр. '+(i+1), cells:cells });
  }
  txt(s, 'Внутрішній цикл обирає гіперпараметри. Зовнішній дає оцінку всієї процедури — разом із підбором. Повторити для кожного зовнішнього фолду.',
    { x:M+0.24, y:4.1, w:6.52, h:0.7, fontSize:11.5, color:C.ink2 });
  card(s, { x:M+7.25, y:1.62, w:CW-7.25, h:1.6, tint:C.alarmT, border:C.alarm, title:'Та сама метафора патрона',
    body:['Якщо ви тим самим фолдом обрали гіперпараметри і ним же відзвітували — це витік. Просто повільний і непомітний.'] });
  card(s, { x:M+7.25, y:3.42, w:CW-7.25, h:1.6, tint:C.panel, title:'Коли цим можна не займатися',
    body:['Якщо є окремий, жодного разу не торканий тест — nested CV не потрібен. Він потрібен тоді, коли даних мало й окремий тест ви дозволити собі не можете.'] });
};
