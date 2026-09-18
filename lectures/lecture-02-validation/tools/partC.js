const D = require('./deck');
const { C, F, W, M, CW, slide, txt, card, foldRow, legend, flowStep, arrow, lead, src, chip, bullets, runs } = D;

module.exports = function(pres){
  const C3 = (CW - 0.56)/3, C2 = (CW - 0.32)/2, C4 = (CW - 0.72)/4;
  let s;

  // 28 ── precision vs базова частота
  s = slide({ eyebrow:'Блок 4 · незбалансовані класи · 14 хв', titleSize:25, titleH:0.95, title:'Precision залежить від базової частоти, а не лише від моделі',
    notes:'Це фундаментальна межа, відома в безпеці з 1999 року як base-rate fallacy: у задачах виявлення саме частота хибних тривог, а не чутливість, обмежує придатність системи (Axelsson, ACM TISSEC 2000). У сучасних оглядах робіт із безпеки цю помилку допускають 77 % авторів.' });
  s.addChart(pres.ChartType.line, [
    { name:'FPR 5 % — поточна модель', labels:['0,1 %','0,5 %','1 %','2 %','5 %','10 %','20 %','50 %'], values:[0.018,0.083,0.154,0.269,0.486,0.667,0.818,0.947] },
    { name:'FPR 0,9 % — потрібна модель', labels:['0,1 %','0,5 %','1 %','2 %','5 %','10 %','20 %','50 %'], values:[0.090,0.332,0.500,0.669,0.839,0.917,0.961,0.990] }
  ], Object.assign({}, D.chartBase, { x:M, y:1.6, w:7.0, h:4.1, chartColors:[C.alarm, C.ok], lineSize:2.5,
      lineDataSymbol:'circle', lineDataSymbolSize:6, showLegend:true, legendPos:'b',
      legendColor:C.muted, legendFontFace:F.sans, legendFontSize:10,
      valAxisMinVal:0, valAxisMaxVal:1, valAxisTitle:'precision (recall зафіксовано на 0,90)', showValAxisTitle:true,
      valAxisTitleColor:C.muted, valAxisTitleFontSize:10, valAxisTitleFontFace:F.sans,
      catAxisTitle:'базова частота цілей', showCatAxisTitle:true,
      catAxisTitleColor:C.muted, catAxisTitleFontSize:10, catAxisTitleFontFace:F.sans }));
  card(s, { x:M+7.25, y:1.6, w:CW-7.25, h:2.4, tint:C.panel, title:'Ключове питання',
    body:['Якою має бути FPR, щоб при тому самому recall 0,90 precision дорівнював 0,5?',
      {t:' ',br:1},{t:'≈ 0,9 %',m:1,b:1,c:C.ok,sz:17,br:1},{t:' ',br:1},
      'Тобто ', {t:'у 5,5 раза менше',b:1}, ', ніж було. Та сама модель із трохи іншим порогом цього не дасть — потрібна інша модель або інший сенсорний контур.'] });
  card(s, { x:M+7.25, y:4.18, w:CW-7.25, h:1.9, tint:C.alarmT, border:C.alarm, title:'Наслідок, про який забувають',
    body:['Базова частота на полігоні й у застосуванні різна. ', {t:'Precision змінюється навіть за незмінної моделі.',b:1},
      ' Модель, перевірена на датасеті з 20 % цілей, у полі з 0,5 % цілей поводиться зовсім інакше.'] });

  // 29 ── інструменти
  s = slide({ eyebrow:'Блок 4 · інструменти', title:'Інструменти й межі їхньої застосовності',
    notes:'Порядок дій на практиці: спершу поріг, потім ваги, і тільки якщо це не спрацювало — ресемплінг. Студенти роблять навпаки.' });
  card(s, { x:M, y:1.62, w:C4, h:3.0, tint:C.white, border:C.line, title:'Ваги класів / cost-sensitive loss',
    titleSize:13, titleH:0.5, body:['Найпростіше і найчесніше: задати вартість помилок явно, а не через маніпуляції з вибіркою.',
      {t:' ',br:1},{t:'руйнує калібрування',m:1,c:C.alarm,sz:10}], size:11 });
  card(s, { x:M+(C4+0.24), y:1.62, w:C4, h:3.0, tint:C.white, border:C.line, title:'Ресемплінг: under / over / SMOTE',
    titleSize:13, titleH:0.5, body:['SMOTE у високій розмірності інтерполює в майже порожньому просторі; на категоріальних ознаках породжує неіснуючі значення; ',
      {t:'до розбиття — це витік',b:1,c:C.alarm}, ' (кейс 4).',
      {t:' ',br:1},{t:'руйнує калібрування',m:1,c:C.alarm,sz:10}], size:11 });
  card(s, { x:M+2*(C4+0.24), y:1.62, w:C4, h:3.0, tint:C.okT, border:C.ok, title:'Підбір порога',
    titleSize:13, titleH:0.5, body:['Найдешевший і найчастіше забутий інструмент. Модель дає ', {t:'ранжування',b:1}, ', рішення дає ', {t:'поріг',b:1}, '.',
      {t:' ',br:1},{t:' ',br:1},
      'Поріг — продуктова, а не статистична величина: його диктує вартість пропуску проти вартості хибної тривоги і пропускна здатність оператора.'], size:11 });
  card(s, { x:M+3*(C4+0.24), y:1.62, w:C4, h:3.0, tint:C.white, border:C.line, title:'Коли позитивів майже немає',
    titleSize:13, titleH:0.5, body:['Переформулювати як виявлення аномалій: вчимося на нормі, шукаємо відхилення.',
      {t:' ',br:1},{t:'міст до лекції 14',m:1,c:C.muted,sz:10}], size:11 });
  lead(s, ['Порядок дій: спершу ', {t:'поріг',b:1}, ', потім ', {t:'ваги',b:1}, ', і лише якщо не спрацювало — ресемплінг. Студенти роблять навпаки.'], { y:5.05, size:14 });

  // 30 ── ROC vs PR
  s = slide({ eyebrow:'Блок 4 · метрики', title:'ROC-AUC 0.98 і PR-AUC 0.65 — це одна й та сама модель',
    notes:'Не казати «ROC-AUC поганий». Казати: ROC-AUC відповідає на питання про ранжування і не залежить від базової частоти, PR-AUC — на питання про роботу оператора і залежить. Для рідкісних цілей вам потрібне друге.' });
  s.addChart(pres.ChartType.scatter, [
    { name:'X-Axis', values:[0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1] },
    { name:'ROC', values:[0,0.436,0.638,0.727,0.809,0.901,0.950,0.982,0.998,1] }
  ], Object.assign({}, D.chartBase, { x:M, y:1.62, w:5.85, h:3.5, chartColors:[C.train],
      lineSize:2.5, lineDataSymbol:'none', showLegend:false,
      valAxisMinVal:0, valAxisMaxVal:1, catAxisMinVal:0, catAxisMaxVal:1,
      valAxisTitle:'recall', showValAxisTitle:true, valAxisTitleColor:C.muted, valAxisTitleFontSize:10, valAxisTitleFontFace:F.sans,
      catAxisTitle:'FPR', showCatAxisTitle:true, catAxisTitleColor:C.muted, catAxisTitleFontSize:10, catAxisTitleFontFace:F.sans }));
  txt(s, 'ROC-AUC ≈ 0,98 — не залежить від базової частоти', { x:M, y:5.22, w:5.85, h:0.3, fontSize:11.5, bold:true, color:C.train, align:'center' });
  s.addChart(pres.ChartType.scatter, [
    { name:'X-Axis', values:[0.05,0.215,0.436,0.638,0.727,0.809,0.901,0.950,0.982,1] },
    { name:'PR', values:[0.99,0.956,0.815,0.563,0.423,0.290,0.154,0.088,0.047,0.01] }
  ], Object.assign({}, D.chartBase, { x:M+6.25, y:1.62, w:5.85, h:3.5, chartColors:[C.alarm],
      lineSize:2.5, lineDataSymbol:'none', showLegend:false,
      valAxisMinVal:0, valAxisMaxVal:1, catAxisMinVal:0, catAxisMaxVal:1,
      valAxisTitle:'precision', showValAxisTitle:true, valAxisTitleColor:C.muted, valAxisTitleFontSize:10, valAxisTitleFontFace:F.sans,
      catAxisTitle:'recall', showCatAxisTitle:true, catAxisTitleColor:C.muted, catAxisTitleFontSize:10, catAxisTitleFontFace:F.sans }));
  txt(s, 'PR-AUC ≈ 0,65 при базовій лінії 0,01 — залежить', { x:M+6.25, y:5.22, w:5.85, h:0.3, fontSize:11.5, bold:true, color:C.alarm, align:'center' });
  lead(s, ['Робоча точка одна й та сама: recall 90 % при FPR 5 %. ', {t:'ROC-AUC оптимістичний при сильному перекосі',b:1},
    ', бо FPR ділиться на величезну кількість негативів: 495 хибних тривог — це «всього 5 %». PR-крива рахує ту саму помилку відносно 100 позитивів і показує 15 %.'], { y:5.78, size:13.5, h:0.9 });

  // 31 ── метрики робочої точки
  s = slide({ eyebrow:'Блок 4 · метрики', title:'Метрики, які фіксують робочу точку',
    notes:'У протидії БпЛА поріг диктує не статистика, а оператор: система, налаштована ловити все, генерує стільки нуйсенс-тривог, що реальні починають ігнорувати. Промислова відповідь — підняття порога плюс підтвердження другим сенсором.' });
  card(s, { x:M, y:1.62, w:C4, h:3.1, tint:C.white, border:C.line, title:'recall @ фіксована FPR / FPPI',
    titleSize:13, titleH:0.5, body:['Галузевий стандарт у виявленні: спершу фіксуємо допустимий потік хибних тривог, потім міряємо, скільки цілей знайдено.',
      {t:' ',br:1},{t:'Pd = 0.875 @ FAR 5 %',m:1,b:1,c:C.ink,br:1},{t:'Pd = 0.745 @ FAR 1 %',m:1,b:1,c:C.ink,br:1},
      {t:'arXiv 2606.29589',m:1,c:C.muted,sz:9}], size:11 });
  card(s, { x:M+(C4+0.24), y:1.62, w:C4, h:3.1, tint:C.trainT, border:C.train, title:'Те саме у світі LLM-агентів',
    titleSize:13, titleH:0.5, body:['Детектор промпт-ін\'єкцій оцінюють саме так, бо доброякісного трафіку в тисячі разів більше за атаки:',
      {t:' ',br:1},{t:'94,8 %',m:1,b:1,c:C.train}, ' атак @ FPR 1 %', {t:' ',br:1},
      {t:'87,8 %',m:1,b:1,c:C.train}, ' @ FPR 0,5 %', {t:' ',br:1},
      {t:'65,3 %',m:1,b:1,c:C.train}, ' @ FPR 0,1 %', {t:' ',br:1},
      {t:'PromptShield, arXiv 2501.15145',m:1,c:C.muted,sz:9}], size:11 });
  card(s, { x:M+2*(C4+0.24), y:1.62, w:C4, h:3.1, tint:C.white, border:C.line, title:'precision @ k',
    titleSize:13, titleH:0.5, body:['Коли оператор фізично перевірить лише k об\'єктів за зміну, усе, що нижче k у ранжуванні, не існує. Міряйте те, що він реально побачить.'], size:11 });
  card(s, { x:M+3*(C4+0.24), y:1.62, w:C4, h:3.1, tint:C.white, border:C.line, title:'MCC, balanced accuracy',
    titleSize:13, titleH:0.5, body:['Коли треба одне число для порівняння моделей і обидва класи важливі. MCC стійкіший до перекосу, ніж F1.'], size:11 });
  lead(s, ['Зверніть увагу на форму запису: ', {t:'метрика без названої робочої точки — це не результат',b:1,c:C.alarm}, '. «Recall 0.9» без FPR не означає нічого.'], { y:5.15, size:14 });

  // 32 ── калібрування: означення
  s = slide({ eyebrow:'Блок 5 · калібрування ймовірностей · 10 хв', title:'Калібрована ймовірність: означення на пальцях',
    notes:'Розклад Brier проговорити: reliability — це калібрування, resolution — здатність розділяти класи, uncertainty — властивість задачі. Калібрування покращує перший доданок і не чіпає другий.' });
  lead(s, ['Серед усіх випадків, де модель сказала ', {t:'0,8',m:1,b:1}, ', позитивних має бути приблизно ', {t:'80 %',b:1}, '. Не 95 і не 60.'], { y:1.5, size:14.5 });
  s.addChart(pres.ChartType.scatter, [
    { name:'X-Axis', values:[0.35,0.45,0.55,0.65,0.75,0.85,0.95] },
    { name:'ідеал',  values:[0.35,0.45,0.55,0.65,0.75,0.85,0.95] },
    { name:'до калібрування',  values:[0.28,0.35,0.42,0.48,0.55,0.64,0.78] },
    { name:'після temperature scaling', values:[0.33,0.44,0.53,0.64,0.73,0.84,0.93] }
  ], Object.assign({}, D.chartBase, { x:M, y:2.15, w:7.0, h:3.6, chartColors:[C.muted, C.alarm, C.ok],
      lineSize:2.5, lineDataSymbol:'circle', lineDataSymbolSize:6, showLegend:true, legendPos:'b',
      legendColor:C.muted, legendFontFace:F.sans, legendFontSize:10,
      valAxisMinVal:0.2, valAxisMaxVal:1, catAxisMinVal:0.3, catAxisMaxVal:1,
      valAxisTitle:'спостережувана частка позитивів', showValAxisTitle:true,
      valAxisTitleColor:C.muted, valAxisTitleFontSize:10, valAxisTitleFontFace:F.sans,
      catAxisTitle:'впевненість, яку заявила модель', showCatAxisTitle:true,
      catAxisTitleColor:C.muted, catAxisTitleFontSize:10, catAxisTitleFontFace:F.sans }));
  card(s, { x:M+7.25, y:2.15, w:CW-7.25, h:1.6, tint:C.alarmT, border:C.alarm, title:'Крива нижче діагоналі',
    body:['= ', {t:'самовпевненість',b:1}, '. Модель заявляє 0,9, а правий лише в 64 % випадків. ECE ≈ 0,13 до і ≈ 0,02 після.'] });
  card(s, { x:M+7.25, y:3.95, w:CW-7.25, h:1.8, tint:C.white, border:C.line, title:'Чим міряємо',
    body:[{t:'ECE',m:1,b:1,c:C.ink}, ' середній зважений розрив по бінах', {t:' ',br:1},
      {t:'Brier',m:1,b:1,c:C.ink}, ' reliability − resolution + uncertainty', {t:' ',br:1},
      {t:'log-loss',m:1,b:1,c:C.ink}, ' жорстко карає впевнені помилки'] });

  // 33 ── навіщо в автономних системах
  s = slide({ eyebrow:'Блок 5 · калібрування ймовірностей', title:'Навіщо це в автономних системах',
    notes:'Місток для тих, хто робить агентів: будь-який поріг «ескалювати людині / діяти самостійно» стоїть на впевненості моделі. Якщо вона некалібрована, ваш поріг означає не те, що ви думаєте.' });
  card(s, { x:M, y:1.62, w:C2, h:2.45, tint:C.white, border:C.line, title:'Score не закінчується на моделі', bullets:[
    'входить у трекер як ймовірність існування об\'єкта',
    'входить у фільтр як вага вимірювання',
    'входить у правило рішення за очікуваною вартістю',
    'входить у злиття кількох сенсорів як апостеріорна ймовірність'], size:12 });
  txt(s, 'Некалібрований score ламає все, що стоїть після моделі — мовчки і без помилки в логах.',
    { x:M+0.24, y:4.16, w:C2-0.48, h:0.45, fontSize:12, bold:true, color:C.alarm });
  card(s, { x:M+C2+0.32, y:1.62, w:C2, h:2.85, tint:C.panel, border:C.line, title:'Типові патології',
    body:[{t:'наївний Баєс',b:1,c:C.ink}, ' — крайня самовпевненість через припущення незалежності', {t:' ',br:1},
      {t:'сучасні нейромережі',b:1,c:C.ink}, ' — самовпевнені тим сильніше, чим більша ємність', {t:' ',br:1},
      {t:'SVM',b:1,c:C.ink}, ' — відступи не є ймовірностями; сигмоїдна форма', {t:' ',br:1},
      {t:'випадковий ліс',b:1,c:C.ink}, ' — «боїться» країв: усереднення не дає ні 0, ні 1'], size:12 });
  card(s, { x:M, y:4.7, w:CW, h:1.5, tint:C.trainT, border:C.train, title:'Той самий ефект у LLM',
    body:['У технічному звіті GPT-4 базова модель на підвибірці MMLU калібрована майже ідеально (', {t:'ECE ≈ 0,007',m:1,b:1},
      '), а після RLHF ECE зростає приблизно ', {t:'на порядок',b:1}, ' (', {t:'≈ 0,074',m:1,b:1},
      '). Модель навчилася звучати впевнено — і саме тому її logprob більше не можна класти в правило прийняття рішення без перекалібрування.'], size:12.5 });

  // 34 ── три методи
  s = slide({ eyebrow:'Блок 5 · калібрування ймовірностей', title:'Три методи і коли який',
    notes:'У scikit-learn це CalibratedClassifierCV з method=\'sigmoid\' або \'isotonic\' і cv=5. Підкреслити: вона робить крос-валідацію саме для того, щоб не калібрувати на train.' });
  const mt = [
    ['Platt scaling','Сигмоїда з двома параметрами поверх score. Мало даних, параметрична форма, стійкий.','< 1 000 прикладів калібрування'],
    ['Ізотонічна регресія','Непараметрична монотонна функція. Гнучкіша, але схильна до перенавчання на малих вибірках.','> кількох тисяч прикладів'],
    ['Temperature scaling','Один параметр на всі логіти. Не змінює argmax, тому accuracy лишається тією самою.','багатокласові мережі']
  ];
  mt.forEach(function(t,i){
    card(s, { x:M+i*(C3+0.28), y:1.62, w:C3, h:2.0, tint:C.white, border:C.line, title:t[0], titleSize:13.5, titleH:0.35,
      body:[t[1], {t:' ',br:1},{t:t[2],m:1,c:C.accent,sz:10}], size:11.5 });
  });
  card(s, { x:M, y:3.85, w:C2, h:1.95, tint:C.alarmT, border:C.alarm, title:'Застереження 1',
    body:['Калібрувати ', {t:'на окремій вибірці або через CV',b:1}, ', ніколи на train. Калібрування на train дає ідеальну діаграму надійності й нульову користь.'] });
  card(s, { x:M+C2+0.32, y:3.85, w:C2, h:1.95, tint:C.alarmT, border:C.alarm, title:'Застереження 2',
    body:['Усі три методи ', {t:'монотонні',b:1}, '. Тому AUC не змінюється взагалі — а всі порогові метрики й рішення за вартістю змінюються. Хто звітує «після калібрування AUC не виріс, отже, не допомогло» — не зрозумів, що робив.'] });

  // 35 ── місток 4 → 5
  s = slide({ eyebrow:'Блок 5 · місток між блоками 4 і 5', title:'Ваги класів і ресемплінг руйнують калібрування',
    notes:'Точка, де сходяться блоки 4 і 5. Студенти вчать їх як окремі теми і не бачать конфлікту: перемогли дисбаланс — зламали ймовірності.' });
  card(s, { x:M, y:1.7, w:7.0, h:2.8, tint:C.alarmT, border:C.alarm, title:'Чому',
    body:['Модель, навчена на збалансованій вибірці або з вагами класів, вчиться під ', {t:'штучну апріорну ймовірність',b:1},
      '. Вона оцінює P(ціль | x) для світу, де цілей 50 %, а не 1 %.',
      {t:' ',br:1},{t:' ',br:1},
      'Її виходи більше не є ймовірностями у вашому світі — навіть якщо recall і precision виглядають прийнятно.'], size:12.5 });
  card(s, { x:M+7.25, y:1.7, w:CW-7.25, h:1.75, tint:C.okT, border:C.ok, title:'Що робити', bullets:[
    'поправка на апріорну ймовірність через шанси',
    'або перекалібрування на вибірці з природною пропорцією',
    'і в обох випадках — заново підібрати поріг'], size:11 });
  card(s, { x:M+7.25, y:3.6, w:CW-7.25, h:1.1, tint:C.panel,
    body:['Якщо ви застосували ваги або ресемплінг і після цього використовуєте predict_proba для чогось, крім ранжування — у вас помилка.'], size:11.5 });

  // 36 ── офлайн vs цільова
  s = slide({ eyebrow:'Блок 6 · офлайнова vs цільова метрика · 5 хв', title:'Офлайнова метрика — це проксі, і в нього є точки розриву',
    notes:'Це РН6, він повертається в лекції 12. Тут лише словник і причини розбіжності. Не заглиблюватися — 5 хвилин.' });
  const fx = [M, M+2.55, M+5.1, M+7.65, M+10.2];
  const fwid = 2.15;
  ['метрика на кадрі','агрегація до об\'єкта','агрегація до місії','рішення оператора','цільова метрика'].forEach(function(t,i){
    flowStep(s, { x:fx[i], y:1.6, w:fwid, h:0.5, t:t, size:10,
      fill:i===4?C.accent:C.white, color:i===4?C.white:C.ink2, border:i===4?C.accent:C.line });
    if (i<4) arrow(s, fx[i]+fwid+0.06, 1.64);
  });
  const gp = [
    ['Зсув розподілу','Полігон не дорівнює театру дій: інша місцевість, погода, пора року, тактика.'],
    ['Замкнена петля','Модель впливає на власні входи: оператор дивиться туди, куди вона вказала, і розмічає те, що вона знайшла.'],
    ['Рівень агрегації','Кадр vs об\'єкт vs місія. Пропустити ціль у 9 кадрах із 10 і знайти в одному — успіх на рівні місії й провал на рівні кадру.'],
    ['Асиметрія вартості й людина в петлі','Оператор ігнорує систему, яка дає 5 000 хибних тривог на годину. Її фактичний recall — нуль.']
  ];
  gp.forEach(function(t,i){
    card(s, { x:M+i*(C4+0.24), y:2.45, w:C4, h:2.35, tint:C.white, border:C.line,
      tag:'розрив', tagFill:C.alarmT, tagColor:C.alarm, tagW:1.0, title:t[0], titleSize:12.5, titleH:0.55, body:[t[1]], size:10.5 });
  });
  lead(s, ['Правило: спершу формулюємо ', {t:'цільову метрику',b:1,c:C.accent},
    ' (хибних тривог на годину польоту, частка підтверджених цілей, час оператора до виявлення), і тільки потім обираємо офлайновий проксі — та перевіряємо, чи вони взагалі корелюють.'], { y:5.05, size:13.5, h:0.8 });
  src(s, 'Те саме в бенчмарках LLM-агентів: аудит 17 бенчмарків знайшов помилки оцінювання до 100 % у відносному вираженні — arXiv 2507.02825', { y:5.98 });

  // 37 ── чек-ліст
  s = slide({ eyebrow:'Блок 7 · чек-ліст · 3 хв', title:'Чек-ліст, за яким я прийматиму захисти',
    notes:'Роздатковий матеріал на одну сторінку. Роздати саме тут, у кінці, щоб під час лекції не читали замість слухання.' });
  const ck = [
    'Яке прикладне рішення ухвалюється і яка цільова метрика?',
    'Що є одиницею незалежності? Яка група?',
    'Схема розбиття і чому саме вона.',
    'Тест заморожено. Скільки разів на нього дивилися?',
    'Увесь препроцесинг усередині Pipeline, усередині фолду.',
    'Метрика доречна класовому перекосу, наведено довірчий інтервал.',
    'Поріг обрано усвідомлено, під вартість помилок.',
    'Ймовірності перевірено на калібрування.',
    'Названо очікувану причину розходження з полем.'
  ];
  ck.forEach(function(t,i){
    const col = i < 5 ? 0 : 1, row = i < 5 ? i : i-5;
    const x = M + col*(C2+0.32), y = 1.62 + row*0.78;
    s.addShape('ellipse', { x:x, y:y+0.06, w:0.44, h:0.44, fill:{color:C.panel}, line:{color:C.line,width:1} });
    txt(s, String(i+1), { x:x, y:y+0.06, w:0.44, h:0.44, fontSize:13, bold:true, color:C.accent, fontFace:F.mono, align:'center', valign:'middle' });
    txt(s, t, { x:x+0.6, y:y, w:C2-0.7, h:0.58, fontSize:12.5, color:C.ink2, valign:'middle' });
  });
  lead(s, ['Дев\'ять питань. На кожне має бути відповідь одним реченням — у звіті, до захисту, а не на захисті.'], { y:5.7, size:14 });

  // 38 ── анонс + гачок
  s = slide({ eyebrow:'Блок 7 · анонс · 3 хв', title:'ЛР2 і гачок на лекцію 3',
    notes:'Закінчити рівно на цьому. Питання — після, у перерві або в чаті курсу.' });
  card(s, { x:M, y:1.7, w:C2, h:2.6, tint:C.white, border:C.line, title:'ЛР2 · дерева рішень + метод головних компонент', titleSize:14, titleH:0.5,
    bullets:['PCA лише всередині Pipeline','групове розбиття, група названа явно в звіті',
      'метрики з довірчими інтервалами','поріг обґрунтовано вартістю помилок'], size:12 });
  txt(s, 'Хто зробить PCA до розбиття — отримає завищений результат і мінус на захисті.',
    { x:M+0.24, y:4.48, w:C2-0.48, h:0.55, fontSize:12.5, bold:true, color:C.alarm });
  txt(s, 'Захист ЛР1 — 24.09', { x:M+0.24, y:5.15, w:C2-0.48, h:0.3, fontSize:11, color:C.muted, fontFace:F.mono });
  card(s, { x:M+C2+0.32, y:1.7, w:C2, h:3.6, tint:C.dark, border:C.dark, title:'Лекція 3', titleColor:C.accentL, titleSize:16,
    body:[{t:'Ми навчилися ',c:'C7CFD8'},{t:'чесно міряти',b:1,c:C.white},{t:' помилку.',c:'C7CFD8',br:1},{t:' ',br:1},
      {t:'Наступного разу — звідки вона береться і чому не можна одночасно мінімізувати зсув і дисперсію.',c:'C7CFD8'}], size:15 });

  // 39 ── джерела
  s = slide({ eyebrow:'Додаток', title:'Джерела прикладів',
    notes:'Числа на слайдах наведені так, як їх подають першоджерела. Частина робіт — препринти, тож перед лекцією варто звірити свіжі версії.' });
  const srcL = [
    ['Витік і відтворюваність', [
      'S. Kapoor, A. Narayanan. Leakage and the reproducibility crisis in ML-based science. Patterns, 2023',
      'D. Arp et al. Dos and Don\'ts of Machine Learning in Computer Security. USENIX Security, 2022',
      'S. Axelsson. The base-rate fallacy and the difficulty of intrusion detection. ACM TISSEC, 2000']],
    ['Групове й часове розбиття у виявленні БпЛА', [
      'How Much Do RF Drone Benchmarks Overstate? arXiv:2607.01025',
      'EchoHawk: acoustic pipeline & session-level leakage. arXiv:2606.29589',
      'Beyond the Performance Illusion (VisDrone, SASP). arXiv:2607.02055',
      'F. Pendlebury et al. TESSERACT. USENIX Security, 2019']]
  ];
  const srcR = [
    ['Хибні кореляції в ATR', [
      'Discovering and Explaining the Non-Causality of Deep Learning in SAR ATR. arXiv:2304.00668',
      'Contrastive Feature Alignment for Clutter Robust SAR Recognition. arXiv:2304.01747',
      'Gwern. The Neural Net Tank Urban Legend — gwern.net/tank']],
    ['Контамінація, калібрування, агенти', [
      'The SWE-Bench Illusion. arXiv:2506.12286',
      'H. He, спостереження про Codeforces, 2023',
      'OpenAI. GPT-4 Technical Report, 2023 — калібрування до/після RLHF',
      'PromptShield: Deployable Detection for Prompt Injection. arXiv:2501.15145',
      'Establishing Best Practices for Building Rigorous Agentic Benchmarks. arXiv:2507.02825']]
  ];
  function srcCol(items, x){
    let y = 1.62;
    items.forEach(function(g){
      txt(s, g[0], { x:x, y:y, w:C2, h:0.3, fontSize:12.5, bold:true, color:C.accent });
      y += 0.36;
      g[1].forEach(function(l){
        txt(s, '·  ' + l, { x:x, y:y, w:C2, h:0.42, fontSize:10, color:C.ink2, fontFace:F.mono, lineSpacingMultiple:1.15 });
        y += 0.42;
      });
      y += 0.18;
    });
  }
  srcCol(srcL, M);
  srcCol(srcR, M+C2+0.32);
};
