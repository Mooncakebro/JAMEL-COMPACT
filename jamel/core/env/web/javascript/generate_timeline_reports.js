const MCR = require('monocart-coverage-reports');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto'); // ⬅️ 引入内置的加密模块

const main = async () => {
    let inputFiles = process.argv.slice(2);
    if (inputFiles.length === 0) {
        console.error("❌ 错误：请提供至少一个输入 JSON 文件的路径。");
        process.exit(1);
    }

    inputFiles = inputFiles.filter(file => fs.existsSync(file)).sort((a, b) => {
        return path.basename(path.dirname(a)).localeCompare(path.basename(path.dirname(b))); 
    });

    const trendData = [];
    const istanbulDataList = [];

    console.log(`🔍 检测到 ${inputFiles.length} 个 coverage 文件，开始生成趋势报告...\n`);

    for (let i = 0; i < inputFiles.length; i++) {
        const currentFile = inputFiles[i];
        const currentFolder = path.basename(path.dirname(currentFile));
        
        console.log(`==================================================`);
        console.log(`⚙️  正在处理单步数据: [${currentFolder}]`);

        const v8Data = JSON.parse(fs.readFileSync(currentFile, 'utf-8'));
        if (v8Data.length == 0) {
            console.warn(`no coverage data found in ${currentFile}! skip.`)
            continue;
        }
        
        // ==============================================================
        // 核心修复：处理超长 URL，防止 Windows 文件名溢出 (MAX_PATH 限制)
        // ==============================================================
        v8Data.forEach(entry => {
            if (!entry.url) return;
            
            // 先去除 hash 锚点
            let urlClean = entry.url.split('#')[0];
            const parts = urlClean.split('?');
            
            if (parts.length > 1) {
                const basePath = parts[0];
                const query = parts.slice(1).join('?');
                
                // 如果参数长度超过 30，将其转换为 8 位的短 Hash
                if (query.length > 30) {
                    const hash = crypto.createHash('md5').update(query).digest('hex').substring(0, 8);
                    // 用 _q_ 连接，既保持了唯一性，又避免了非法字符和路径超长
                    entry.url = `${basePath}_q_${hash}`;
                } else {
                    entry.url = urlClean;
                }
            }
        });
        // ==============================================================
        const tempOutputDir = `./data/temp/${currentFolder}`;
        
        const tempMcr = MCR({
            outputDir: tempOutputDir,
            reports: ['json'], 
            entryFilter: (entry) => entry.url.includes('http')
        });
        await tempMcr.add(v8Data);
        await tempMcr.generate();

        const istanbulPath = path.join(tempOutputDir, 'coverage-final.json');
        const istanbulData = JSON.parse(fs.readFileSync(istanbulPath, 'utf-8'));
        istanbulDataList.push(istanbulData);

        const outputDir = `./data/coverage-report/${currentFolder}`;
        console.log(`📈 正在生成累计报告 (Step ${i + 1}/${inputFiles.length})`);
        
        const mcrAccumulated = MCR({
            name: `Accumulated Coverage (${currentFolder})`,
            outputDir: outputDir,
            reports: ['html', 'console-summary', 'json-summary'], 
        });

        for (let j = 0; j <= i; j++) {
            await mcrAccumulated.add(istanbulDataList[j]);
        }

        const results = await mcrAccumulated.generate();
        const stats = results.summary.total ? results.summary.total : results.summary;
        
        if (stats && stats.lines) {
            trendData.push({
                step: i + 1,
                timestamp_folder: currentFolder,
                statements_covered: stats.statements.covered,
                statements_pct: stats.statements.pct,
                lines_covered: stats.lines.covered,
                lines_pct: stats.lines.pct,
                branches_covered: stats.branches.covered,
                branches_pct: stats.branches.pct
            });
            console.log(`  📊 当前累计指标 -> 分支数: ${stats.branches.covered} | 语句数: ${stats.statements.covered}`);
        }
    }
    
    const trendFilePath = './data/coverage-report/coverage_growth_trend.json';
    fs.writeFileSync(trendFilePath, JSON.stringify(trendData, null, 2), 'utf-8');
    
    console.log(`\n🎉 趋势报告生成完毕！`);
    console.log(`📉 趋势数据已保存至: ${trendFilePath}`);
};

main();