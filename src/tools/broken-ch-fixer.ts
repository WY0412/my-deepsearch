import { generateText } from "ai";
import { getModel } from "../config";
import { TrackerContext } from "../types";
import { detectBrokenUnicodeViaFileIO } from "../utils/text-tools";
import { logInfo, logError, logDebug, logWarning } from '../logging';


/**
 * Repairs markdown content with characters by using Gemini to guess the missing text
 */
export async function repairUnknownChars(mdContent: string, trackers?: TrackerContext): Promise<string> {
  const { broken, readStr } = await detectBrokenUnicodeViaFileIO(mdContent);
  if (!broken) return readStr;
  logWarning("Detected broken unicode in output, attempting to repair...");

  let repairedContent = readStr;
  let remainingUnknowns = true;
  let iterations = 0;

  let lastPosition = -1;

  while (remainingUnknowns && iterations < 20) {
    iterations++;

    // Find the position of the first � character
    const position = repairedContent.indexOf('�');
    if (position === -1) {
      remainingUnknowns = false;
      continue;
    }

    // Check if we're stuck at the same position
    if (position === lastPosition) {
      // Move past this character by removing it
      repairedContent = repairedContent.substring(0, position) +
        repairedContent.substring(position + 1);
      continue;
    }

    // Update last position to detect loops
    lastPosition = position;

    // Count consecutive � characters
    let unknownCount = 0;
    for (let i = position; i < repairedContent.length && repairedContent[i] === '�'; i++) {
      unknownCount++;
    }

    // Extract context around the unknown characters
    const contextSize = 50;
    const start = Math.max(0, position - contextSize);
    const end = Math.min(repairedContent.length, position + unknownCount + contextSize);
    const leftContext = repairedContent.substring(start, position);
    const rightContext = repairedContent.substring(position + unknownCount, end);

    // Ask Gemini to guess the missing characters
    try {
      const result = await generateText({
        model: getModel('fallback'),
        system: `您正在帮助修复一个包含污点（由表示）的受损扫描markdown文档。
通过查看周围的上下文，确定符号处应该是什么原始文本。

规则：
1. 仅输出准确的替换文本 - 不要解释、引用或添加额外文本
2. 保持您的回应与未知序列长度相适应
3. 如果上下文表明文档可能是中文，请考虑这一点`,
        prompt: `
受损文本中连续出现了${unknownCount}个符号。

污点左侧内容："${leftContext}"
污点右侧内容："${rightContext}"

这两段上下文之间的原始文本是什么？`,
      });

      trackers?.tokenTracker.trackUsage('md-fixer', result.usage)
      const replacement = result.text.trim();

      // Validate the replacement
      if (
        replacement === "UNKNOWN" ||
        (await detectBrokenUnicodeViaFileIO(replacement)).broken ||
        replacement.length > unknownCount * 4
      ) {
        logWarning(`Skipping invalid replacement ${replacement} at position ${position}`);
        // Skip to the next character without modifying content
      } else {
        // Replace the unknown sequence with the generated text
        repairedContent = repairedContent.substring(0, position) +
          replacement +
          repairedContent.substring(position + unknownCount);
      }

      logDebug(`Repair iteration ${iterations}: replaced ${unknownCount} chars with "${replacement}"`);

    } catch (error) {
      logError("Error repairing unknown characters:", { error });
      // Skip to the next character without modifying this one
    }
  }

  return repairedContent;
}