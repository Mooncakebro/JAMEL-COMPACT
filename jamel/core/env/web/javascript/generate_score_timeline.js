const MCR = require('monocart-coverage-reports');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const coverageScore = (summary) => {
  const stats = summary.total ? summary.total : summary;
  return ['lines', 'branches', 'statements', 'functions'].reduce((acc, key) => {
    return acc + ((stats[key] && stats[key].covered) || 0);
  }, 0);
};

const normalizeV8Data = (v8Data) => {
  v8Data.forEach((entry) => {
    if (!entry.url) return;
    let urlClean = entry.url.split('#')[0];
    const parts = urlClean.split('?');
    if (parts.length > 1) {
      const basePath = parts[0];
      const query = parts.slice(1).join('?');
      if (query.length > 30) {
        const hash = crypto.createHash('md5').update(query).digest('hex').substring(0, 8);
        entry.url = basePath + '_q_' + hash;
      } else {
        entry.url = urlClean;
      }
    }
  });
  return v8Data;
};

const main = async () => {
  const inputFiles = process.argv.slice(2).filter((file) => fs.existsSync(file));
  if (inputFiles.length === 0) {
    console.error('Usage: node generate_score_timeline.js coverage1.json coverage2.json ...');
    process.exit(1);
  }
  const tempRoot = './data/temp/score_timeline';
  fs.rmSync(tempRoot, { recursive: true, force: true });
  fs.mkdirSync(tempRoot, { recursive: true });
  const istanbulDataList = [];
  const rows = [];
  let previousScore = 0;
  for (let i = 0; i < inputFiles.length; i++) {
    const v8Data = normalizeV8Data(JSON.parse(fs.readFileSync(inputFiles[i], 'utf-8')));
    const convDir = path.join(tempRoot, 'conv_' + i);
    const tempMcr = MCR({ outputDir: convDir, reports: ['json'], entryFilter: (entry) => entry.url.includes('http') });
    await tempMcr.add(v8Data);
    await tempMcr.generate();
    const istanbulPath = path.join(convDir, 'coverage-final.json');
    if (fs.existsSync(istanbulPath)) {
      istanbulDataList.push(JSON.parse(fs.readFileSync(istanbulPath, "utf-8")));
    }
    const accDir = path.join(tempRoot, 'acc_' + i);
    const accMcr = MCR({ outputDir: accDir, reports: ['json-summary'] });
    for (const istanbulData of istanbulDataList) await accMcr.add(istanbulData);
    const results = await accMcr.generate();
    const currentScore = coverageScore(results.summary);
    rows.push({ index: i, file: inputFiles[i], previous_score: previousScore, current_score: currentScore, delta_score: currentScore - previousScore });
    previousScore = currentScore;
  }
  console.log(JSON.stringify(rows));
};

main().catch((error) => { console.error(error); process.exit(1); });
