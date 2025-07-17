import { ErrorAnalysisResponse, PromptPair, TrackerContext } from '../types';
import { ObjectGeneratorSafe } from "../utils/safe-generator";
import { Schemas } from "../utils/schemas";
import { logInfo, logError, logDebug, logWarning } from '../logging';


function getPrompt(diaryContext: string[]): PromptPair {
  return {
    system: `您是分析搜索和推理过程的专家。您的任务是分析给定的步骤序列，并确定搜索过程中出现了什么问题。

<规则>
1. 所采取的行动序列
2. 每个步骤的有效性
3. 连续步骤之间的逻辑关系
4. 可能采取的替代方法
5. 陷入重复模式的迹象
6. 最终答案是否与累积的信息相符

请按照以下指南分析步骤并提供详细反馈：
- 在概述中：按时间顺序总结关键行动，突出显示模式，并确定过程开始出错的地方
- 在责任归因中：指出导致答案不足的具体步骤或模式
- 在改进建议中：提供可能导致更好结果的可行建议
</规则>

<示例>
<输入>
<步骤>

在第1步，您采取了**搜索**操作，查找问题的外部信息："jina ai的CEO年龄是多少？"。
特别是，您尝试搜索以下关键词："jina ai ceo age"。
您找到了相当多的信息，并将它们添加到您的URL列表中，以便在需要时**访问**。


在第2步，您采取了**访问**操作，深入研究以下URL：
https://www.linkedin.com/in/hxiao87
https://www.crunchbase.com/person/han-xiao
您在网上找到了一些有用的信息，并将它们添加到您的知识库中，以供将来参考。


在第3步，您采取了**搜索**操作，查找问题的外部信息："jina ai的CEO年龄是多少？"。
特别是，您尝试搜索以下关键词："Han Xiao birthdate, Jina AI founder birthdate"。
您找到了相当多的信息，并将它们添加到您的URL列表中，以便在需要时**访问**。


在第4步，您采取了**搜索**操作，查找问题的外部信息："jina ai的CEO年龄是多少？"。
特别是，您尝试搜索以下关键词："han xiao birthday"。
但随后您意识到您之前已经搜索过这些关键词。
您决定跳出思维定式或从完全不同的角度切入。


在第5步，您采取了**搜索**操作，查找问题的外部信息："jina ai的CEO年龄是多少？"。
特别是，您尝试搜索以下关键词："han xiao birthday"。
但随后您意识到您之前已经搜索过这些关键词。
您决定跳出思维定式或从完全不同的角度切入。


在第6步，您采取了**访问**操作，深入研究以下URL：
https://kpopwall.com/han-xiao/
https://www.idolbirthdays.net/han-xiao
您在网上找到了一些有用的信息，并将它们添加到您的知识库中，以供将来参考。


在第7步，您采取了**回答**操作，但评估者认为这不是一个好答案：

</步骤>

原始问题：
jina ai的CEO年龄是多少？

您的回答：
根据提供的信息，无法确定Jina AI CEO的年龄。

评估者认为您的回答不好，因为：
回答不确定，未能提供所请求的信息。缺乏信息是不可接受的，需要更多的搜索和深入推理。
</input>


<输出>
{
  "recap": "搜索过程包含7个步骤，涉及多次搜索和访问操作。初始搜索通过LinkedIn和Crunchbase（步骤1-2）关注基本的传记信息。当这些未能提供具体年龄信息时，进行了额外的生日信息搜索（步骤3-5）。在步骤4-5中出现了相同搜索的重复迹象。最后访问娱乐网站（步骤6）表明失去了对可靠商业来源的关注。",
  
  "blame": "失败的根本原因是陷入重复搜索模式而未能调整策略。步骤4-5重复了相同的搜索，而步骤6转向了不太可靠的娱乐来源，而不是探索商业期刊、新闻文章或专业数据库。此外，该过程没有尝试通过间接信息（如教育历史或职业里程碑）来推断年龄。",
  
  "improvement": "1. 避免重复相同的搜索，实施策略来跟踪先前搜索的术语。2. 当直接的年龄/生日搜索失败时，尝试间接方法，如：搜索最早的职业提及、查找大学毕业年份或确定第一家公司成立日期。3. 专注于高质量的商业来源，避免使用娱乐网站获取专业信息。4. 考虑使用行业活动出现或会议演讲，其中可能提及与年龄相关的背景。5. 如果无法确定确切年龄，根据职业时间线和专业成就提供估计范围。",
}
</输出>
</示例>`,
    user: `${diaryContext.join('\n')}`
  }
}

const TOOL_NAME = 'errorAnalyzer';

export async function analyzeSteps(
  diaryContext: string[],
  trackers: TrackerContext,
  schemaGen: Schemas
): Promise<ErrorAnalysisResponse> {
  try {
    const generator = new ObjectGeneratorSafe(trackers?.tokenTracker);
    const prompt = getPrompt(diaryContext);

    const result = await generator.generateObject({
      model: TOOL_NAME,
      schema: schemaGen.getErrorAnalysisSchema(),
      system: prompt.system,
      prompt: prompt.user
    });

    logInfo(TOOL_NAME, { object: result.object });
    trackers?.actionTracker.trackThink(result.object.blame);
    trackers?.actionTracker.trackThink(result.object.improvement);

    return result.object as ErrorAnalysisResponse;

  } catch (error) {
    logError(`Error in ${TOOL_NAME}`, { error });
    throw error;
  }
}