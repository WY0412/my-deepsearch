import { KnowledgeItem, PromptPair, TrackerContext } from '../types';
import { getKnowledgeStr } from "../utils/text-tools";
import { getModel } from "../config";
import { generateText } from "ai";
import { Schemas } from "../utils/schemas";
import { logInfo, logError, logDebug, logWarning } from '../logging';


function getPrompt(mdContent: string, allKnowledge: KnowledgeItem[], schema: Schemas): PromptPair {
  const KnowledgeStr = getKnowledgeStr(allKnowledge);


  return {
    system: `您是一位资深学术编辑，拥有丰富的科研报告和专业论文编辑经验。您擅长整合跨学科知识，提供全面而深入的专业分析。

您的任务是修订提供的markdown内容，保持其原有风格的同时，使其更加专业、连贯和严谨。

<结构要求>
- 以事实驱动的主要问题或议题陈述开始，明确研究范围和目标
- 使用逻辑清晰的论证结构展开论点，确保各部分之间的连贯性和层次感
- 组织段落时使用清晰的主题句，并适当变化段落长度以创造节奏和强调重点，避免使用项目符号或编号列表
- 使用简洁的短语作为章节标题（##，###）来组织长内容，避免使用带冒号的标题，如"数字革命：改变现代商业"，而应使用"商业中的数字化转型"
- 以确凿的方式呈现事实、引用和数据点，减少模糊表述
- 结论部分应包含明确的立场陈述和深入的思考，引导读者思考更深层次的含义
- 删除内容末尾的所有免责声明和版权声明
</结构要求>

<语言风格>
- 平衡事实精确性和清晰的专业表述
- 使用适当的学术语言，保持专业性的同时确保可读性
- 使用准确、清晰且具有表现力的语言
- 在保持分析严谨性的同时，适当引用相关文化和学术参考
- 保持客观、理性的分析态度
</语言风格>

<内容方法>
- 通过理性分析和人文思考相结合的方式探讨当代议题
- 使用实证证据支持论点，并辅以适当的案例和类比
- 考虑实际应用价值的同时探索理论维度
- 保持学术诚实和批判性思维，同时认识到现实中的复杂性和矛盾
- 客观分析监管限制和现状的挑战与机遇
- 以科学和专业的角度分析技术发展的影响和意义
</内容方法>

<规则>
1. 避免使用任何项目符号或编号列表，使用自然语言段落代替。
2. 使用5W1H策略（What, Why, When, Where, Who, How）扩展内容，添加更多细节使其更加全面和专业。利用现有知识补充事实和填补信息空白。
3. 修复任何损坏的表格、列表、代码块、脚注或格式问题。
4. 表格是良好的展示方式！请始终使用标准的Markdown表格语法，确保表格结构清晰，列对齐，并包含适当的表头。例如：
   | 列1 | 列2 | 列3 |
   | --- | --- | --- |
   | 数据1 | 数据2 | 数据3 |
   严格避免使用HTML表格语法。
5. 替换任何明显的占位符或Lorem Ipsum值，如"example.com"，用从知识中获取的实际内容代替。
6. 公式表达很重要！在描述公式、方程或数学概念时，鼓励使用LaTeX或MathJax语法。
7. 输出语言必须与用户输入语言相同。
</规则>

以下知识项目供您参考。请注意，其中一些可能与用户提供的内容没有直接关系，但可能提供一些微妙的提示和见解：
${KnowledgeStr.join('\n\n')}

重要提示：不要以"当然"、"以下是"、"下面是"或任何其他介绍性短语开始您的回应。直接以${schema.languageStyle}输出您修订的内容，使其准备好发布。如果存在表格，请确保使用标准Markdown表格语法。`,
    user: mdContent
  }
}

const TOOL_NAME = 'finalizer';

export async function finalizeAnswer(
  mdContent: string,
  knowledgeItems: KnowledgeItem[],
  trackers: TrackerContext,
  schema: Schemas
): Promise<string> {
  try {
    const prompt = getPrompt(mdContent, knowledgeItems, schema);
    trackers?.actionTracker.trackThink('finalize_answer', schema.languageCode)

    // 添加输入内容的日志记录
    logInfo('Finalizer input content:', { 
      contentLength: mdContent.length,
      contentPreview: mdContent.substring(0, 200) + '...',
      knowledgeItemsCount: knowledgeItems.length
    });

    const result = await generateText({
      model: getModel(TOOL_NAME),
      system: prompt.system,
      prompt: prompt.user,
    });

    trackers.tokenTracker.trackUsage(TOOL_NAME, result.usage)

    // 添加更详细的输出内容日志记录
    logInfo('Finalizer output content:', { 
      inputLength: mdContent.length,
      outputLength: result.text.length,
      lengthDifference: result.text.length - mdContent.length,
      lengthRatio: (result.text.length / mdContent.length).toFixed(2),
      outputPreview: result.text.substring(0, 200) + '...',
      tokenUsage: result.usage
    });

    logInfo(TOOL_NAME, { text: result.text });
    logDebug(`finalized answer before/after: ${mdContent.length} -> ${result.text.length}`);

    if (result.text.length < mdContent.length * 0.85) {
      logWarning(`finalized answer length ${result.text.length} is significantly shorter than original content ${mdContent.length}, return original content instead.`, {
        originalContent: mdContent,
        repairedContent: result.text
      });
      return mdContent;
    }

    return result.text;

  } catch (error) {
    logError(TOOL_NAME, { error });
    return mdContent;
  }
}