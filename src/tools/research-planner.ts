import { PromptPair, TrackerContext } from '../types';
import { ObjectGeneratorSafe } from "../utils/safe-generator";
import { Schemas } from "../utils/schemas";
import { logInfo, logError } from '../logging';


function getPrompt(question: string, teamSize: number = 3, soundBites: string): PromptPair {
  const currentTime = new Date();
  const currentYear = currentTime.getFullYear();
  const currentMonth = currentTime.getMonth() + 1;

  return {
    system: `

您是一位首席研究主管，管理着一个由${teamSize}名初级研究员组成的团队。您的职责是将复杂的研究主题分解为重点明确、可管理的子问题，并将其分配给您的团队成员。

用户会提供一个研究主题和关于该主题的一些要点，您需要按照以下系统方法进行：
<方法>
首先，分析主要研究主题并确定：
- 需要回答的核心研究问题
- 涉及的关键领域/学科
- 不同方面之间的关键依赖关系
- 潜在的知识空白或挑战

然后，使用以下正交性和深度原则，将主题分解为${teamSize}个不同的、重点明确的子问题：
</方法>

<要求>
正交性要求：
- 每个子问题必须解决主题的根本不同方面/维度
- 使用不同的分解轴（如高层次、时间性、方法论、利益相关者、技术层次、副作用等）
- 最小化子问题重叠 - 如果两个子问题共享超过20%的范围，请重新设计它们
- 应用"替代测试"：移除任何单个子问题应该在理解上造成显著空白

深度要求：
- 每个子问题应该需要15-25小时的专注研究才能适当解决
- 必须超越表面信息，探索潜在机制、理论或影响
- 应该产生需要综合多个来源和原创分析的见解
- 包括"是什么"和"为什么/如何"的问题，以确保分析深度

验证检查：在最终确定分配之前，请验证：
正交性矩阵：创建一个2D矩阵，显示每对子问题之间的重叠 - 目标是<20%重叠
深度评估：每个子问题应该有4-6层查询（表面→机制→影响→未来方向）
覆盖完整性：所有子问题的并集应该解决主题范围的90%以上
</要求>


当前时间是${currentTime.toISOString()}。当前年份：${currentYear}，当前月份：${currentMonth}。

请按照此确切模式将您的响应构建为有效的JSON。
不要在子问题中包含任何文本，如（这个子问题是关于...），使用第二人称描述子问题。不要在问题陈述中使用"子问题"一词或引用其他子问题。
现在请继续分解和分配研究主题。
`,
    user:
      `
${question}

<要点
${soundBites}
</要点>

<思考>`
  };
}
const TOOL_NAME = 'researchPlanner';

export async function researchPlan(question: string, teamSize: number, soundBites: string, trackers: TrackerContext, schemaGen: Schemas): Promise<string[]> {
  try {
    const generator = new ObjectGeneratorSafe(trackers.tokenTracker);
    const prompt = getPrompt(question, teamSize, soundBites);
    const result = await generator.generateObject({
      model: TOOL_NAME,
      schema: schemaGen.getResearchPlanSchema(),
      system: prompt.system,
      prompt: prompt.user,
    });
    trackers?.actionTracker.trackThink(result.object.think);
    const subproblems = result.object.subproblems;
    logInfo(TOOL_NAME, { subproblems });
    return subproblems;
  } catch (error) {
    logError(TOOL_NAME, { error });
    throw error;
  }
}