const MCR = require('monocart-coverage-reports');
const fs = require('fs');
const path = require('path');

const parseArgs = () => {
    const args = process.argv.slice(2);
    const inputFiles = [];
    let outputDir = null;

    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        if (arg === '--output-dir') {
            outputDir = args[i + 1];
            i += 1;
            continue;
        }
        inputFiles.push(arg);
    }

    return { inputFiles, outputDir };
};

const main = async () => {
    // 1. 获取命令行参数中所有的文件路径
    const { inputFiles, outputDir: requestedOutputDir } = parseArgs();

    if (inputFiles.length === 0) {
        console.error("❌ 错误：请提供至少一个输入 JSON 文件的路径！");
        console.error("用法：node generate_cumulative_report.js [--output-dir dir] file1.json file2.json dir/file3.json ...");
        process.exit(1);
    }

    // 获取最后一个文件的名字（不带后缀）作为输出目录名
    const lastFilePath = inputFiles[inputFiles.length - 1];
    const outputDir = requestedOutputDir || `./data/coverage-report/${path.parse(lastFilePath).name}`;

    console.log(`🔍 共检测到 ${inputFiles.length} 个输入文件`);
    console.log(`📂 最终输出目录: ${outputDir}\n`);

    const istanbulDataList = [];

    // =========================================================================
    // 阶段 1：将每一个独立的 V8 数据转换为 Istanbul 格式（防止后续合并时字节偏移冲突）
    // =========================================================================
    console.log("⚙️  [阶段 1/2] 正在将 V8 数据安全转换为 Istanbul 格式...");

    for (let i = 0; i < inputFiles.length; i++) {
        const filePath = inputFiles[i];
        
        if (!fs.existsSync(filePath)) {
            console.warn(`⚠️ 警告: 文件未找到，跳过 -> ${filePath}`);
            continue;
        }

        console.log(`  ⏳ 转换单步数据: ${filePath}`);
        const reportData = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        
        // 创建一个唯一的临时目录
        const tempRoot = './data/temp/generate_report';
        fs.mkdirSync(tempRoot, { recursive: true });
        const tempOutputDir = fs.mkdtempSync(path.join(tempRoot, 'conv_'));
        
        const tempMcr = MCR({
            outputDir: tempOutputDir,
            reports: ['json'], // 仅生成 coverage-final.json (Istanbul 格式)
            entryFilter: (entry) => entry.url.includes('http')
        });

        await tempMcr.add(reportData);
        await tempMcr.generate();

        // 读取转换好的 Istanbul 数据并存入内存
        const istanbulPath = path.join(tempOutputDir, 'coverage-final.json');
        if (fs.existsSync(istanbulPath)) {
            const istanbulData = JSON.parse(fs.readFileSync(istanbulPath, 'utf-8'));
            istanbulDataList.push(istanbulData);
        }

        // 清理临时目录，保持磁盘整洁
        fs.rmSync(tempOutputDir, { recursive: true, force: true });
    }

    // =========================================================================
    // 阶段 2：将所有绝对安全的 Istanbul 数据进行最终的累加合并
    // =========================================================================
    console.log("\n📈 [阶段 2/2] 正在合并所有格式化后的数据...");

    // 注意：因为输入已经是 Istanbul 格式，可视化报告我们使用标准的 'html'
    const mcr = MCR({
        name: 'Cumulative Python Playwright Coverage Report (Istanbul)',
        outputDir: outputDir, 
        reports: [
            'html',             // 经典的按目录/文件层级展示的网页报告
            'console-summary',  // 控制台打印汇总表格
            'json-summary'      // 供机器读取的精简 JSON 数据
        ]
    });

    // 循环把内存中所有的 Istanbul 数据合并在一起
    for (const istanbulData of istanbulDataList) {
        await mcr.add(istanbulData);
    }

    console.log("正在生成最终的 HTML 报告...");

    // 一次性生成最终的报告
    await mcr.generate();
    
    console.log(`\n✅ 累计报告生成完毕！`);
    console.log(`👉 请打开浏览器查看: ${outputDir}/index.html`);
};

main();
