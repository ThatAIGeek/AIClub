const PptxGenJS = require('pptxgenjs');
const D = require('./deck');
const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';
pres.author = 'Курс машинного навчання';
pres.title = 'Лекція 2. Валідація моделей та методологія експерименту';
pres.subject = 'Валідація, крос-валідація, витік даних, дисбаланс класів, калібрування';
D.init(pres);
require('./partA')(pres);
require('./partB')(pres);
require('./partC')(pres);
const out = process.argv[2] || 'lecture-02.pptx';
pres.writeFile({ fileName: out }).then(function(f){ console.log('written', f, '| slides:', D.count()); })
  .catch(function(e){ console.error('ERROR', e); process.exit(1); });
