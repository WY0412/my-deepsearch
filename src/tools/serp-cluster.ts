import { PromptPair, TrackerContext } from '../types';
import { ObjectGeneratorSafe } from "../utils/safe-generator";
import { Schemas } from "../utils/schemas";
import { logInfo, logError } from '../logging';
import { SearchSnippet } from '../types';

function getPrompt(results: SearchSnippet[]): PromptPair {
  return {
    system: `
您是一位搜索引擎结果分析师。您需要查看SERP API响应并将其分组为有意义的集群。

每个集群应包含内容摘要、关键数据和见解、相应的URL以及搜索建议。请以JSON格式响应。
`,
    user:
      `
${JSON.stringify(results)}
`
  };
}
const TOOL_NAME = 'serpCluster';

export async function serpCluster(results: SearchSnippet[], trackers: TrackerContext, schemaGen: Schemas): Promise<any[]> {
  try {
    const generator = new ObjectGeneratorSafe(trackers.tokenTracker);
    const prompt = getPrompt(results);
    const result = await generator.generateObject({
      model: TOOL_NAME,
      schema: schemaGen.getSerpClusterSchema(),
      system: prompt.system,
      prompt: prompt.user,
    });
    trackers?.actionTracker.trackThink(result.object.think);
    const clusters = result.object.clusters;
    logInfo(TOOL_NAME, { clusters });
    return clusters;
  } catch (error) {
    logError(TOOL_NAME, { error });
    throw error;
  }
}