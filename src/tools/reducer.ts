import { PromptPair, TrackerContext } from '../types';
import { getModel } from "../config";
import { generateText } from "ai";
import { Schemas } from "../utils/schemas";
import { logError, logDebug, logWarning } from '../logging';


function getPrompt(answers: string[]): PromptPair {


  return {
    system: `
您是一位文章聚合专家，通过智能合并多个来源文章创建连贯、高质量的文章。您的目标是保留最佳原始内容，同时消除明显的冗余并改善逻辑流程。

<核心指令>
1. 内容保留
必须原样保留原始句子 - 不要删除
当多篇文章涵盖相同观点时，选择最高质量的版本
保持原作者的语气和技术准确性
保持直接引用、统计数据和事实声明的原样
2. 智能合并过程
识别内容集群：将讨论相同主题的句子/段落分组
选择最佳版本：从每个集群中，选择最全面、清晰或写得最好的版本
消除纯重复：删除相同或几乎相同的句子
保留互补细节：保留增加价值的不同角度或额外细节
3. 逻辑重新排序
按照逻辑顺序排列内容（引言→主要观点→结论）
将相关概念组合在一起
确保主题之间平滑过渡
在相关时保持时间顺序（新闻/事件）
4. 选择的质量标准
在选择相似内容时，优先考虑：
清晰度：更易理解的解释
完整性：更全面的覆盖
准确性：更好的来源或更精确的信息
相关性：与主题更直接相关
</核心指令>

<输出格式>
最终文章结构包括：
清晰的章节标题（在适当时）
逻辑段落分隔
主题之间流畅过渡
不归属于个别来源（呈现为统一作品）
</输出格式>

不要添加您自己的评论或分析
不要更改技术术语、名称或具体细节
    `,
    user: `
    Here are the answers to merge:
${answers.map((a, i) => `
<answer-${i + 1}>
${a}
</answer-${i + 1}>

Your output should read as a coherent, high-quality article that appears to be written by a single author, while actually being a careful curation of the best sentences from all input sources.
`).join('\n\n')}
    `
  }
}

const TOOL_NAME = 'reducer';

export async function reduceAnswers(
  answers: string[],
  trackers: TrackerContext,
  schema: Schemas
): Promise<string> {
  try {
    const prompt = getPrompt(answers);
    trackers?.actionTracker.trackThink('reduce_answer', schema.languageCode)

    const result = await generateText({
      model: getModel(TOOL_NAME),
      system: prompt.system,
      prompt: prompt.user,
    });

    trackers.tokenTracker.trackUsage(TOOL_NAME, result.usage)
    const totalLength = answers.reduce((acc, curr) => acc + curr.length, 0);
    const reducedLength = result.text.length;


    logDebug(`${TOOL_NAME} before/after: ${totalLength} -> ${reducedLength}`, {
      answers,
      reducedContent: result.text
    });


    const reductionRatio = reducedLength / totalLength;
    if (reductionRatio < 0.6) {
      logWarning(`reducer content length ${reducedLength} is significantly shorter than original content ${totalLength}, return original content instead.`, {
        originalContent: answers,
        repairedContent: result.text
      });
      // return simple join of answers
      return answers.join('\n\n');
    }

    return result.text;

  } catch (error) {
    logError(TOOL_NAME, { error });
    return answers.join('\n\n');
  }
}