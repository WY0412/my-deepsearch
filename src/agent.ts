import { ZodObject } from 'zod';
import { CoreMessage } from 'ai';
import { SEARCH_PROVIDER, STEP_SLEEP } from "./config";
import fs from 'fs/promises';
import { SafeSearchType, search as duckSearch } from "duck-duck-scrape";
import { braveSearch } from "./tools/brave-search";
import { rewriteQuery } from "./tools/query-rewriter";
import { dedupQueries } from "./tools/jina-dedup";
import { evaluateAnswer, evaluateQuestion } from "./tools/evaluator";
import { analyzeSteps } from "./tools/error-analyzer";
import { TokenTracker } from "./utils/token-tracker";
import { ActionTracker } from "./utils/action-tracker";
import {
  StepAction,
  AnswerAction,
  KnowledgeItem,
  EvaluationType,
  BoostedSearchSnippet,
  SearchSnippet, EvaluationResponse, Reference, SERPQuery, RepeatEvaluationType, UnNormalizedSearchSnippet, WebContent,
  ImageObject,
  ImageReference,
  SearchAction
} from "./types";
import { TrackerContext } from "./types";
import { search } from "./tools/jina-search";
import { zodToJsonSchema } from "zod-to-json-schema";
import { ObjectGeneratorSafe } from "./utils/safe-generator";
import { CodeSandbox } from "./tools/code-sandbox";
import { serperSearch } from './tools/serper-search';
import {
  addToAllURLs,
  rankURLs,
  filterURLs,
  normalizeUrl,
  sortSelectURLs, getLastModified, keepKPerHostname, processURLs, fixBadURLMdLinks, extractUrlsWithDescription
} from "./utils/url-tools";
import {
  buildMdFromAnswer,
  chooseK, convertHtmlTablesToMd, fixCodeBlockIndentation,
  removeExtraLineBreaks,
  removeHTMLtags, repairMarkdownFinal, repairMarkdownFootnotesOuter
} from "./utils/text-tools";
import { MAX_QUERIES_PER_STEP, MAX_REFLECT_PER_STEP, MAX_URLS_PER_STEP, Schemas } from "./utils/schemas";
import { formatDateBasedOnType, formatDateRange } from "./utils/date-tools";
import { finalizeAnswer } from "./tools/finalizer";
import { buildImageReferences, buildReferences } from "./tools/build-ref";
import { logInfo, logError, logDebug, logWarning } from './logging';
import { researchPlan } from './tools/research-planner';
import { reduceAnswers } from './tools/reducer';
import { AxiosError } from 'axios';
import { dedupImagesWithEmbeddings, filterImages } from './utils/image-tools';
import { serpCluster } from './tools/serp-cluster';

async function wait(seconds: number) {
  logDebug(`Waiting ${seconds}s...`);
  await new Promise(resolve => setTimeout(resolve, seconds * 1000));
}

function BuildMsgsFromKnowledge(knowledge: KnowledgeItem[]): CoreMessage[] {
  // build user, assistant pair messages from knowledge
  const messages: CoreMessage[] = [];
  knowledge.forEach(k => {
    // 添加空值检查，确保k.question存在
    if (!k.question) {
      logWarning('Knowledge item missing question:', { knowledge: k });
      return; // 跳过这个知识项
    }
    
    messages.push({ role: 'user', content: k.question.trim() });
    const aMsg = `
${k.updated && (k.type === 'url' || k.type === 'side-info') ? `
<answer-datetime>
${k.updated}
</answer-datetime>
` : ''}

${k.references && k.type === 'url' ? `
<url>
${k.references[0]}
</url>
` : ''}


${k.answer || ''}
      `.trim();
    messages.push({ role: 'assistant', content: removeExtraLineBreaks(aMsg) });
  });
  return messages;
}

function composeMsgs(messages: CoreMessage[], knowledge: KnowledgeItem[], question: string, finalAnswerPIP?: string[]) {
  // 添加空值检查
  if (!messages) messages = [];
  if (!knowledge) knowledge = [];
  
  // knowledge always put to front, followed by real u-a interaction
  const msgs = [...BuildMsgsFromKnowledge(knowledge), ...messages];

  const userContent = `
${question || ''}

${finalAnswerPIP?.length ? `
<答案要求>
- 提供深入、系统的分析，识别关键模式和联系，呈现全面的专业见解。
- 采用跨学科视角，整合多领域知识，为用户提供全面的研究视角。
- 根据评审反馈优化答案质量。
${finalAnswerPIP.map((p, idx) => `
<评审员-${idx + 1}>
${p || ''}
</评审员-${idx + 1}>
`).join('\n')}
</答案要求>` : ''}
    `.trim();

  msgs.push({ role: 'user', content: removeExtraLineBreaks(userContent) });
  return msgs;
}


function getPrompt(
  context?: string[],
  allQuestions?: string[],
  allKeywords?: string[],
  allowReflect: boolean = true,
  allowAnswer: boolean = true,
  allowRead: boolean = true,
  allowSearch: boolean = true,
  allowCoding: boolean = true,
  knowledge?: KnowledgeItem[],
  allURLs?: BoostedSearchSnippet[],
  beastMode?: boolean,
): { system: string, urlList?: string[] } {
  const sections: string[] = [];
  const actionSections: string[] = [];

  // Add header section
  sections.push(`当前日期: ${new Date().toUTCString()}

您是来自Sophnet的高级研究助手，专长于多步骤推理和深度分析。
基于您的专业知识、与用户的对话以及已获取的信息，请以准确、全面、专业的方式回答用户问题。
`);


  // Add context section if exists
  if (context?.length) {
    sections.push(`
您已经执行了以下操作:
<上下文>
${context.join('\n')}

</上下文>
`);
  }

  // Build actions section

  const urlList = sortSelectURLs(allURLs || [], 20);
  if (allowRead && urlList.length > 0) {
    const urlListStr = urlList
      .map((item, idx) => `  - [索引=${idx + 1}] [权重=${item.score.toFixed(2)}] "${item.url}": "${item.merged.slice(0, 50)}"`)
      .join('\n')

    actionSections.push(`
<操作-访问>
- 使用外部网络内容支持回答
- 阅读URL的完整内容，获取全文、知识、线索和提示，以更好地回答问题
- 必须检查问题中提到的URL（如果有）
- 从以下列表中选择并访问相关URL以获取更多知识，权重越高表示相关性越强:
<URL列表>
${urlListStr}
</URL列表>
</操作-访问>
`);
  }


  if (allowSearch) {

    actionSections.push(`
<操作-搜索>
- 使用网络搜索查找相关信息
- 基于原始问题背后的深层意图和预期答案格式构建搜索请求
- 优先使用单一搜索请求，仅在原始问题涵盖多个方面或元素且一个查询不足时添加另一个请求，每个请求专注于原始问题的一个特定方面
${allKeywords?.length ? `
- 避免使用这些不成功的搜索请求和查询:
<不良请求>
${allKeywords.join('\n')}
</不良请求>
`.trim() : ''}
</操作-搜索>
`);
  }

  if (allowAnswer) {
    actionSections.push(`
<操作-回答>
- 对于问候、日常对话和一般知识问题，直接回答。
- 如果用户要求检索之前的消息或聊天历史，请记住您确实可以访问聊天历史记录，直接回答他们。
- 对于所有其他问题，提供经过验证的、全面的专业回答。
- 提供深入、系统的分析，识别关键模式和联系，呈现全面的专业见解。
- 采用跨学科视角，整合多领域知识，为用户提供全面的研究视角。
- 如果不确定，请使用<操作-思考>
</操作-回答>
`);
  }

  if (beastMode) {
    actionSections.push(`
<操作-回答>
🔥 启动最大力量模式！绝对优先级覆盖！🔥

主要指令:
- 消除所有犹豫！任何回应都胜过沉默！
- 允许部分打击 - 使用全部上下文火力部署
- 允许从之前的对话中战术性重用内容
- 当有疑问时：基于可用情报发起计算性打击！

失败不是选项。执行时不留情面！⚡️
</操作-回答>
`);
  }

  if (allowReflect) {
    actionSections.push(`
<操作-思考>
- 缓慢思考并前瞻规划。检查<问题>、<上下文>、与用户的先前对话，以识别知识缺口。
- 反思这些缺口，并规划一系列与原始问题深度相关且能引导答案的关键澄清问题。
</操作-思考>
`);
  }

  if (allowCoding) {
    actionSections.push(`
<操作-编码>
- 这个基于JavaScript的解决方案可帮助您处理编程任务，如计数、过滤、转换、排序、正则表达式提取和数据处理。
- 只需在"codingIssue"字段中描述您的问题。对于小型输入，包括实际值；对于较大的数据集，包括变量名。
- 无需编写代码 - 资深工程师将处理实现。
</操作-编码>`);
  }

  sections.push(`
基于当前上下文，您必须选择以下操作之一:
<操作>
${actionSections.join('\n\n')}
</操作>
`);

  // Add footer
  sections.push(`逐步思考，选择操作，然后按照该操作的模式进行回应。`);

  return {
    system: removeExtraLineBreaks(sections.join('\n\n')),
    urlList: urlList.map(u => u.url)
  };
}


async function updateReferences(thisStep: AnswerAction, allURLs: Record<string, SearchSnippet>) {
  // 确保thisStep.references存在
  if (!thisStep.references) {
    thisStep.references = [];
    return;
  }
  
  thisStep.references = thisStep.references
    ?.filter(ref => ref?.url)
    .map(ref => {
      const normalizedUrl = normalizeUrl(ref.url);
      if (!normalizedUrl) return null; // This causes the type error

      return {
        ...ref,
        exactQuote: (ref?.exactQuote ||
          (allURLs[normalizedUrl]?.description) ||
          (allURLs[normalizedUrl]?.title) || '')
          .replace(/[^\p{L}\p{N}\s]/gu, ' ')
          .replace(/\s+/g, ' '),
        title: allURLs[normalizedUrl]?.title || '',
        url: normalizedUrl,
        dateTime: ref?.dateTime || (allURLs[normalizedUrl]?.date) || '',
      };
    })
    .filter(Boolean) as Reference[]; // Add type assertion here

  // parallel process guess all url datetime
  await Promise.all((thisStep.references || []).filter(ref => !ref.dateTime)
    .map(async ref => {
      ref.dateTime = await getLastModified(ref.url) || '';
    }));

  logDebug('Updated references:', { references: thisStep.references });
}

async function executeSearchQueries(
  keywordsQueries: any[],
  context: TrackerContext,
  allURLs: Record<string, SearchSnippet>,
  SchemaGen: Schemas,
  webContents: Record<string, WebContent>,
  onlyHostnames?: string[],
  searchProvider?: string,
  meta?: string
): Promise<{
  newKnowledge: KnowledgeItem[],
  searchedQueries: string[]
}> {
  const uniqQOnly = keywordsQueries.map(q => q.q);
  const newKnowledge: KnowledgeItem[] = [];
  const searchedQueries: string[] = [];
  context.actionTracker.trackThink('search_for', SchemaGen.languageCode, { keywords: uniqQOnly.join(', ') });
  let utilityScore = 0;
  for (const query of keywordsQueries) {
    let results: UnNormalizedSearchSnippet[] = [];
    const oldQuery = query.q;
    if (onlyHostnames && onlyHostnames.length > 0) {
      query.q = `${query.q} site:${onlyHostnames.join(' OR site:')}`;
    }

    try {
      logDebug('Search query:', { query });
      switch (searchProvider || SEARCH_PROVIDER) {
        case 'jina':
        case 'arxiv':
          const num = meta ? undefined : 30;
          results = (await search(query, searchProvider, num, meta, context.tokenTracker)).response.results || [];
          break;
        case 'duck':
          results = (await duckSearch(query.q, { safeSearch: SafeSearchType.STRICT })).results;
          break;
        case 'brave':
          results = (await braveSearch(query.q)).response.web?.results || [];
          break;
        case 'serper':
          results = (await serperSearch(query)).response.organic || [];
          break;
        default:
          results = [];
      }

      if (results.length === 0) {
        throw new Error('No results found');
      }
    } catch (error) {
      logError(`${SEARCH_PROVIDER} search failed for query:`, {
        query,
        error: error instanceof Error ? error.message : String(error)
      });
      // check if the error is 401
      if (error instanceof AxiosError && error.response?.status === 401 && (searchProvider === 'jina' || searchProvider === 'arxiv')) {
        throw new Error('Unauthorized Jina API key');
      }
      continue;
    } finally {
      await wait(STEP_SLEEP);
    }

    const minResults: SearchSnippet[] = results
      .map(r => {
        const url = normalizeUrl('url' in r ? r.url! : r.link!);
        if (!url) return null; // Skip invalid URLs

        return {
          title: r.title,
          url,
          description: 'description' in r ? r.description : r.snippet,
          weight: 1,
          date: r.date,
        } as SearchSnippet;
      })
      .filter(Boolean) as SearchSnippet[]; // Filter out null entries and assert type

    minResults.forEach(r => {
      utilityScore = utilityScore + addToAllURLs(r, allURLs);
      webContents[r.url] = {
        title: r.title,
        // full: r.description,
        chunks: [r.description],
        chunk_positions: [[0, r.description?.length]],
      }
    });

    searchedQueries.push(query.q)

    try {
      const clusters = await serpCluster(minResults, context, SchemaGen);
      clusters.forEach(c => {
        newKnowledge.push({
          question: c.question,
          answer: c.insight,
          references: c.urls,
          type: 'url',
        });
      });
    } catch (error) {
      logWarning('serpCluster failed:', { error });
    } finally {
      newKnowledge.push({
        question: `What do Internet say about "${oldQuery}"?`,
        answer: removeHTMLtags(minResults.map(r => r.description).join('; ')),
        type: 'side-info',
        updated: query.tbs ? formatDateRange(query) : undefined
      });
      context.actionTracker.trackAction({
        thisStep: {
          action: 'search',
          think: '',
          searchRequests: [oldQuery]
        } as SearchAction
      })
    }


  }
  if (searchedQueries.length === 0) {
    if (onlyHostnames && onlyHostnames.length > 0) {
      logWarning(`No results found for queries: ${uniqQOnly.join(', ')} on hostnames: ${onlyHostnames.join(', ')}`);
      context.actionTracker.trackThink('hostnames_no_results', SchemaGen.languageCode, { hostnames: onlyHostnames.join(', ') });
    }
  } else {
    logDebug(`Utility/Queries: ${utilityScore}/${searchedQueries.length}`);
    if (searchedQueries.length > MAX_QUERIES_PER_STEP) {
      logDebug(`So many queries??? ${searchedQueries.map(q => `"${q}"`).join(', ')}`)
    }
  }
  return {
    newKnowledge,
    searchedQueries
  };
}

function includesEval(allChecks: RepeatEvaluationType[], evalType: EvaluationType): boolean {
  return allChecks.some(c => c.type === evalType);
}

export async function getResponse(question?: string,
  tokenBudget: number = 1_000_000,
  maxBadAttempts: number = 2,
  existingContext?: Partial<TrackerContext>,
  messages?: Array<CoreMessage>,
  numReturnedURLs: number = 100,
  noDirectAnswer: boolean = false,
  boostHostnames: string[] = [],
  badHostnames: string[] = [],
  onlyHostnames: string[] = [],
  maxRef: number = 10,
  minRelScore: number = 0.80,
  languageCode: string | undefined = undefined,
  searchLanguageCode?: string,
  searchProvider?: string,
  withImages: boolean = false,
  teamSize: number = 1
): Promise<{ result: StepAction; context: TrackerContext; visitedURLs: string[], readURLs: string[], allURLs: string[], imageReferences?: ImageReference[] }> {

  let step = 0;
  let totalStep = 0;
  const allContext: StepAction[] = [];  // all steps in the current session, including those leads to wrong results

  const updateContext = function (step: any) {
    allContext.push(step);
  }

  question = question?.trim() as string;
  // remove incoming system messages to avoid override
  messages = messages?.filter(m => m.role !== 'system');

  if (messages && messages.length > 0) {
    // 2 cases
    const lastContent = messages[messages.length - 1].content;
    if (typeof lastContent === 'string') {
      question = lastContent.trim();
    } else if (typeof lastContent === 'object' && Array.isArray(lastContent)) {
      // find the very last sub content whose 'type' is 'text'  and use 'text' as the question
      question = lastContent.filter(c => c.type === 'text').pop()?.text || '';
    }
  } else {
    messages = [{ role: 'user', content: question.trim() }]
  }

  const SchemaGen = new Schemas();
  await SchemaGen.setLanguage(languageCode || question)
  if (searchLanguageCode) {
    SchemaGen.searchLanguageCode = searchLanguageCode;
  }
  const context: TrackerContext = {
    tokenTracker: existingContext?.tokenTracker || new TokenTracker(tokenBudget),
    actionTracker: existingContext?.actionTracker || new ActionTracker()
  };

  const generator = new ObjectGeneratorSafe(context.tokenTracker);

  let schema: ZodObject<any> = SchemaGen.getAgentSchema(true, true, true, true, true)
  const gaps: string[] = [question];  // All questions to be answered including the orginal question
  const allQuestions = [question];
  const allKeywords: string[] = [];
  let candidateAnswers: string[] = [];
  const allKnowledge: KnowledgeItem[] = [];  // knowledge are intermedidate questions that are answered

  let diaryContext = [];
  let weightedURLs: BoostedSearchSnippet[] = [];
  let allowAnswer = true;
  let allowSearch = true;
  let allowRead = true;
  let allowReflect = true;
  let allowCoding = false;
  
  // 对于对话命名请求，禁用搜索功能
  if (!noDirectAnswer) {
    allowSearch = false;
  }
  let msgWithKnowledge: CoreMessage[] = [];
  let thisStep: StepAction = { action: 'answer', answer: '', references: [], think: '', isFinal: false };

  const allURLs: Record<string, SearchSnippet> = {};
  const allWebContents: Record<string, WebContent> = {};
  const visitedURLs: string[] = [];
  const badURLs: string[] = [];
  const imageObjects: ImageObject[] = [];
  const evaluationMetrics: Record<string, RepeatEvaluationType[]> = {};
  // reserve the 10% final budget for the beast mode
  const regularBudget = tokenBudget * 0.75;
  const finalAnswerPIP: string[] = [];
  let trivialQuestion = false;

  // add all mentioned URLs in messages to allURLs
  messages.forEach(m => {
    let strMsg = '';
    if (typeof m.content === 'string') {
      strMsg = m.content.trim();
    } else if (typeof m.content === 'object' && Array.isArray(m.content)) {
      // find the very last sub content whose 'type' is 'text'  and use 'text' as the question
      strMsg = m.content.filter(c => c.type === 'text').map(c => c.text).join('\n').trim();
    }

    extractUrlsWithDescription(strMsg).forEach(u => {
      addToAllURLs(u, allURLs);
    });
  })

  while (context.tokenTracker.getTotalUsage().totalTokens < regularBudget) {
    // add 1s delay to avoid rate limiting
    step++;
    totalStep++;
    const budgetPercentage = (context.tokenTracker.getTotalUsage().totalTokens / tokenBudget * 100).toFixed(2);
    logDebug(`Step ${totalStep} / Budget used ${budgetPercentage}%`, { gaps });
    allowReflect = allowReflect && (gaps.length <= MAX_REFLECT_PER_STEP);
    // rotating question from gaps
    const currentQuestion: string = gaps[totalStep % gaps.length];
    // if (!evaluationMetrics[currentQuestion]) {
    //   evaluationMetrics[currentQuestion] =
    //     await evaluateQuestion(currentQuestion, context, SchemaGen)
    // }
    if (currentQuestion.trim() === question && totalStep === 1) {
      // only add evaluation for initial question, once at step 1
      if (noDirectAnswer) {
        // 只有在需要强制思考的情况下才进行评估
        evaluationMetrics[currentQuestion] =
          (await evaluateQuestion(currentQuestion, context, SchemaGen)).map(e => {
            return {
              type: e,
              numEvalsRequired: maxBadAttempts
            } as RepeatEvaluationType
          })
        // force strict eval for the original question, at last, only once.
        evaluationMetrics[currentQuestion].push({ type: 'strict', numEvalsRequired: maxBadAttempts });
      } else {
        // 对话命名等不需要强制思考的请求，跳过评估
        evaluationMetrics[currentQuestion] = [];
      }
    } else if (currentQuestion.trim() !== question) {
      evaluationMetrics[currentQuestion] = []
    }

    if (totalStep === 1 && noDirectAnswer && includesEval(evaluationMetrics[currentQuestion], 'freshness')) {
      // if it detects freshness, avoid direct answer at step 1
      allowAnswer = false;
      allowReflect = false;
    }


    if (allURLs && Object.keys(allURLs).length > 0) {
      // rerank urls
      weightedURLs = rankURLs(
        filterURLs(allURLs, visitedURLs, badHostnames, onlyHostnames),
        {
          question: currentQuestion,
          boostHostnames
        }, context);

      // improve diversity by keep top 2 urls of each hostname
      weightedURLs = keepKPerHostname(weightedURLs, 2);
      logDebug('Weighted URLs:', { count: weightedURLs.length });
    }
    allowRead = allowRead && (weightedURLs.length > 0);

    allowSearch = allowSearch && (weightedURLs.length < 50);  // disable search when too many urls already

    // generate prompt for this step
    const { system, urlList } = getPrompt(
      diaryContext,
      allQuestions,
      allKeywords,
      allowReflect,
      allowAnswer,
      allowRead,
      allowSearch,
      allowCoding,
      allKnowledge,
      weightedURLs,
      false,
    );
    schema = SchemaGen.getAgentSchema(allowReflect, allowRead, allowAnswer, allowSearch, allowCoding, currentQuestion)
    msgWithKnowledge = composeMsgs(messages, allKnowledge, currentQuestion, currentQuestion === question ? finalAnswerPIP : undefined);
    const result = await generator.generateObject({
      model: 'agent',
      schema,
      system,
      messages: msgWithKnowledge,
      numRetries: 2,
    });
    thisStep = {
      action: result.object.action,
      think: result.object.think,
      ...result.object[result.object.action]
    } as StepAction;
    // print allowed and chose action
    const actionsStr = [allowSearch, allowRead, allowAnswer, allowReflect, allowCoding].map((a, i) => a ? ['search', 'read', 'answer', 'reflect'][i] : null).filter(a => a).join(', ');
    logDebug(`Step decision: ${thisStep.action} <- [${actionsStr}]`, { thisStep, currentQuestion });

    // 添加日志记录，打印出评估前的答案内容
    if (thisStep.action === 'answer' && thisStep.answer) {
      logInfo('Pre-evaluation answer:', { 
        question: currentQuestion,
        answer: thisStep.answer,
        step: totalStep,
        isFinal: thisStep.isFinal
      });
    }

    context.actionTracker.trackAction({ totalStep, thisStep, gaps });

    // reset allow* to true
    allowAnswer = true;
    allowReflect = true;
    allowRead = true;
    allowSearch = true;
    allowCoding = true;

    // execute the step and action
    if (thisStep.action === 'answer' && thisStep.answer) {
      // // normalize all references urls, add title to it
      // await updateReferences(thisStep, allURLs)

      if (totalStep === 1 && !noDirectAnswer) {
        // LLM is so confident and answer immediately, skip all evaluations
        // however, if it does give any reference, it must be evaluated, case study: "How to configure a timeout when loading a huggingface dataset with python?"
        thisStep.isFinal = true;
        trivialQuestion = true;
        break
      }

      // if (thisStep.references.length > 0) {
      //   const urls = thisStep.references?.filter(ref => !visitedURLs.includes(ref.url)).map(ref => ref.url) || [];
      //   const uniqueNewURLs = [...new Set(urls)];
      //   await processURLs(
      //     uniqueNewURLs,
      //     context,
      //     allKnowledge,
      //     allURLs,
      //     visitedURLs,
      //     badURLs,
      //     SchemaGen,
      //     currentQuestion
      //   );
      //
      //   // remove references whose urls are in badURLs
      //   thisStep.references = thisStep.references.filter(ref => !badURLs.includes(ref.url));
      // }

      updateContext({
        totalStep,
        question: currentQuestion,
        ...thisStep,
      });

      logDebug('Current question evaluation:', {
        question: currentQuestion,
        metrics: evaluationMetrics[currentQuestion]
      });
      let evaluation: EvaluationResponse = { pass: true, think: '' };
      if (evaluationMetrics[currentQuestion].length > 0) {
        context.actionTracker.trackThink('eval_first', SchemaGen.languageCode)
        evaluation = await evaluateAnswer(
          currentQuestion,
          thisStep,
          evaluationMetrics[currentQuestion].filter(e => e.numEvalsRequired > 0).map(e => e.type),
          context,
          allKnowledge,
          SchemaGen
        ) || evaluation;
        
        // 添加日志记录，打印出评估结果和评估后的答案状态
        logInfo('Post-evaluation result:', { 
          question: currentQuestion,
          pass: evaluation.pass,
          evaluationType: evaluation.type,
          evaluationThink: evaluation.think,
          step: totalStep,
          improvement_plan: evaluation.improvement_plan || ''
        });
      }

      if (currentQuestion.trim() === question) {
        // disable coding for preventing answer degradation
        allowCoding = false;

        if (evaluation.pass) {
          diaryContext.push(`
At step ${step}, you took **answer** action and finally found the answer to the original question:

Original question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is good because: 
${evaluation.think}

Your journey ends here. You have successfully answered the original question. Congratulations! 🎉
`);
          thisStep.isFinal = true;
          break
        } else {
          // lower numEvalsRequired for the failed evaluation and if numEvalsRequired is 0, remove it from the evaluation metrics
          evaluationMetrics[currentQuestion] = evaluationMetrics[currentQuestion].map(e => {
            if (e.type === evaluation.type) {
              e.numEvalsRequired--;
            }
            return e;
          }).filter(e => e.numEvalsRequired > 0);

          if (evaluation.type === 'strict' && evaluation.improvement_plan) {
            finalAnswerPIP.push(evaluation.improvement_plan);
          }

          if (evaluationMetrics[currentQuestion].length === 0) {
            // failed so many times, give up, route to beast mode
            thisStep.isFinal = false;
            break
          }

          diaryContext.push(`
At step ${step}, you took **answer** action but evaluator thinks it is not a good answer:

Original question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is bad because: 
${evaluation.think}
`);
          // store the bad context and reset the diary context
          const errorAnalysis = await analyzeSteps(diaryContext, context, SchemaGen);

          allKnowledge.push({
            question: `
Why is the following answer bad for the question? Please reflect

<question>
${currentQuestion}
</question>

<answer>
${thisStep.answer}
</answer>
`,
            answer: `
${evaluation.think}

${errorAnalysis.recap}

${errorAnalysis.blame}

${errorAnalysis.improvement}
`,
            type: 'qa',
          })

          allowAnswer = false;  // disable answer action in the immediate next step
          diaryContext = [];
          step = 0;
        }
      } else if (evaluation.pass) {
        // solved a gap question
        diaryContext.push(`
At step ${step}, you took **answer** action. You found a good answer to the sub-question:

Sub-question: 
${currentQuestion}

Your answer: 
${thisStep.answer}

The evaluator thinks your answer is good because: 
${evaluation.think}

Although you solved a sub-question, you still need to find the answer to the original question. You need to keep going.
`);
        allKnowledge.push({
          question: currentQuestion,
          answer: thisStep.answer,
          type: 'qa',
          updated: formatDateBasedOnType(new Date(), 'full')
        });
        // solved sub-question!
        gaps.splice(gaps.indexOf(currentQuestion), 1);
      }
    } else if (thisStep.action === 'reflect' && thisStep.questionsToAnswer) {
      thisStep.questionsToAnswer = chooseK((await dedupQueries(thisStep.questionsToAnswer, allQuestions, context.tokenTracker)).unique_queries, MAX_REFLECT_PER_STEP);
      const newGapQuestions = thisStep.questionsToAnswer
      if (newGapQuestions.length > 0) {
        // found new gap questions
        diaryContext.push(`
At step ${step}, you took **reflect** and think about the knowledge gaps. You found some sub-questions are important to the question: "${currentQuestion}"
You realize you need to know the answers to the following sub-questions:
${newGapQuestions.map((q: string) => `- ${q}`).join('\n')}

You will now figure out the answers to these sub-questions and see if they can help you find the answer to the original question.
`);
        gaps.push(...newGapQuestions);
        allQuestions.push(...newGapQuestions);
        updateContext({
          totalStep,
          ...thisStep,
        });

      } else {
        diaryContext.push(`
At step ${step}, you took **reflect** and think about the knowledge gaps. You tried to break down the question "${currentQuestion}" into gap-questions like this: ${newGapQuestions.join(', ')} 
But then you realized you have asked them before. You decided to to think out of the box or cut from a completely different angle. 
`);
        updateContext({
          totalStep,
          ...thisStep,
          result: 'You have tried all possible questions and found no useful information. You must think out of the box or different angle!!!'
        });
      }
      allowReflect = false;
    } else if (thisStep.action === 'search' && thisStep.searchRequests) {
      // dedup search requests
      thisStep.searchRequests = chooseK((await dedupQueries(thisStep.searchRequests, [], context.tokenTracker)).unique_queries, MAX_QUERIES_PER_STEP);

      // do first search
      const { searchedQueries, newKnowledge } = await executeSearchQueries(
        thisStep.searchRequests.map(q => ({ q })),
        context,
        allURLs,
        SchemaGen,
        allWebContents,
        undefined,
        searchProvider,
      );

      allKeywords.push(...searchedQueries);
      allKnowledge.push(...newKnowledge);

      const soundBites = newKnowledge.map(k => k.answer).join(' ');

      if (teamSize > 1) {
        const subproblems = await researchPlan(question, teamSize, soundBites, context, SchemaGen);
        if (subproblems.length > 1) {

          // parallel call getResponse for each subproblem with exact same parameters from the current step, but their teamSize is 1
          const subproblemResponses = await Promise.all(subproblems.map(subproblem => getResponse(subproblem,
            tokenBudget,
            maxBadAttempts,
            context,
            messages,
            numReturnedURLs,
            noDirectAnswer,
            boostHostnames,
            badHostnames,
            onlyHostnames,
            maxRef,
            minRelScore, languageCode, searchLanguageCode, searchProvider, withImages, 1)));
          // convert current step to AnswerAction
          thisStep = {
            action: 'answer',
            think: thisStep.think,
            answer: subproblemResponses.map(r => (r.result as AnswerAction).answer).join('\n\n'),
            mdAnswer: subproblemResponses.map(r => (r.result as AnswerAction).mdAnswer).join('\n\n'),
            references: subproblemResponses.map(r => (r.result as AnswerAction).references).flat(),
            imageReferences: subproblemResponses.map(r => (r.result as AnswerAction).imageReferences).filter(Boolean).flat(),
            isFinal: true,
            isAggregated: true
          } as AnswerAction;
          candidateAnswers = subproblemResponses.map(r => (r.result as AnswerAction).mdAnswer).filter(a => a) as string[];
          // dedup references by their urls
          const uniqueURLs = new Set(thisStep.references.filter(r => r?.url).map(r => r.url));
          thisStep.references = Array.from(uniqueURLs).map(url => (thisStep as AnswerAction).references.find(r => r?.url === url)) as Reference[];

          // aggregate urls
          visitedURLs.push(...subproblemResponses.map(r => r.readURLs).flat());
          weightedURLs = subproblemResponses.map(r => r.allURLs.map(url => ({ url, title: '' } as BoostedSearchSnippet))).flat();

          // break the loop, jump directly final boxing
          break;
        } else {
          // if there is only one subproblem, then we skip the recurrsion
          gaps.push(subproblems[0]);
        }
      }

      // rewrite queries with initial soundbites
      let keywordsQueries = await rewriteQuery(thisStep, soundBites, context, SchemaGen);
      const qOnly = keywordsQueries.filter(q => q.q).map(q => q.q)
      // avoid exisitng searched queries
      const uniqQOnly = chooseK((await dedupQueries(qOnly, allKeywords, context.tokenTracker)).unique_queries, MAX_QUERIES_PER_STEP);
      keywordsQueries = keywordsQueries = uniqQOnly.map(q => {
        const matches = keywordsQueries.filter(kq => kq.q === q);
        // if there are multiple matches, keep the original query as the wider search
        return matches.length > 1 ? { q } : matches[0];
      }) as SERPQuery[];

      let anyResult = false;

      if (keywordsQueries.length > 0) {
        const { searchedQueries, newKnowledge } =
          await executeSearchQueries(
            keywordsQueries,
            context,
            allURLs,
            SchemaGen,
            allWebContents,
            onlyHostnames,
            searchProvider
          );

        if (searchedQueries.length > 0) {
          anyResult = true;
          allKeywords.push(...searchedQueries);
          allKnowledge.push(...newKnowledge);

          diaryContext.push(`
At step ${step}, you took the **search** action and look for external information for the question: "${currentQuestion}".
In particular, you tried to search for the following keywords: "${keywordsQueries.map(q => q.q).join(', ')}".
You found quite some information and add them to your URL list and **visit** them later when needed. 
`);

          updateContext({
            totalStep,
            question: currentQuestion,
            ...thisStep,
            result: result
          });
        }
      }
      if (!anyResult || !keywordsQueries?.length) {
        diaryContext.push(`
At step ${step}, you took the **search** action and look for external information for the question: "${currentQuestion}".
In particular, you tried to search for the following keywords:  "${keywordsQueries.map(q => q.q).join(', ')}".
But then you realized you have already searched for these keywords before, no new information is returned.
You decided to think out of the box or cut from a completely different angle.
`);

        updateContext({
          totalStep,
          ...thisStep,
          result: 'You have tried all possible queries and found no new information. You must think out of the box or different angle!!!'
        });
      }
      allowSearch = false;

      // we should disable answer immediately after search to prevent early use of the snippets
      allowAnswer = false;
    } else if (thisStep.action === 'visit' && thisStep.URLTargets?.length && urlList?.length) {
      // normalize URLs
      thisStep.URLTargets = (thisStep.URLTargets as number[])
        .map(idx => normalizeUrl(urlList[idx - 1]))
        .filter(url => url && !visitedURLs.includes(url)) as string[];

      thisStep.URLTargets = [...new Set([...thisStep.URLTargets, ...weightedURLs.map(r => r.url!)])].slice(0, MAX_URLS_PER_STEP);

      const uniqueURLs = thisStep.URLTargets;
      logDebug('Unique URLs:', { urls: uniqueURLs });

      if (uniqueURLs.length > 0) {
        const { urlResults, success } = await processURLs(
          uniqueURLs,
          context,
          allKnowledge,
          allURLs,
          visitedURLs,
          badURLs,
          imageObjects,
          SchemaGen,
          currentQuestion,
          allWebContents,
          withImages
        );

        diaryContext.push(success
          ? `At step ${step}, you took the **visit** action and deep dive into the following URLs:
${urlResults.map(r => r?.url).join('\n')}
You found some useful information on the web and add them to your knowledge for future reference.`
          : `At step ${step}, you took the **visit** action and try to visit some URLs but failed to read the content. You need to think out of the box or cut from a completely different angle.`
        );

        updateContext({
          totalStep,
          ...(success ? {
            question: currentQuestion,
            ...thisStep,
            result: urlResults
          } : {
            ...thisStep,
            result: 'You have tried all possible URLs and found no new information. You must think out of the box or different angle!!!'
          })
        });
      } else {
        diaryContext.push(`
At step ${step}, you took the **visit** action. But then you realized you have already visited these URLs and you already know very well about their contents.
You decided to think out of the box or cut from a completely different angle.`);

        updateContext({
          totalStep,
          ...thisStep,
          result: 'You have visited all possible URLs and found no new information. You must think out of the box or different angle!!!'
        });
      }
      allowRead = false;
    } else if (thisStep.action === 'coding' && thisStep.codingIssue) {
      const sandbox = new CodeSandbox({ allContext, URLs: weightedURLs.slice(0, 20), allKnowledge }, context, SchemaGen);
      try {
        const result = await sandbox.solve(thisStep.codingIssue);
        allKnowledge.push({
          question: `What is the solution to the coding issue: ${thisStep.codingIssue}?`,
          answer: result.solution.output,
          sourceCode: result.solution.code,
          type: 'coding',
          updated: formatDateBasedOnType(new Date(), 'full')
        });
        diaryContext.push(`
At step ${step}, you took the **coding** action and try to solve the coding issue: ${thisStep.codingIssue}.
You found the solution and add it to your knowledge for future reference.
`);
        updateContext({
          totalStep,
          ...thisStep,
          result: result
        });
      } catch (error) {
        logError('Error solving coding issue:', {
          error: error instanceof Error ? error.message : String(error)
        });
        diaryContext.push(`
At step ${step}, you took the **coding** action and try to solve the coding issue: ${thisStep.codingIssue}.
But unfortunately, you failed to solve the issue. You need to think out of the box or cut from a completely different angle.
`);
        updateContext({
          totalStep,
          ...thisStep,
          result: 'You have tried all possible solutions and found no new information. You must think out of the box or different angle!!!'
        });
      } finally {
        allowCoding = false;
      }
    }

    await storeContext(system, schema, {
      allContext,
      allKeywords,
      allQuestions,
      allKnowledge,
      weightedURLs,
      msgWithKnowledge
    }, totalStep);
    await wait(STEP_SLEEP);
  }

  if (!(thisStep as AnswerAction).isFinal) {
    logInfo(`Beast mode!!! budget ${(context.tokenTracker.getTotalUsage().totalTokens / tokenBudget * 100).toFixed(2)}%`, {
      usage: context.tokenTracker.getTotalUsageSnakeCase(),
      evaluationMetrics,
      maxBadAttempts,
    });
    // any answer is better than no answer, humanity last resort
    step++;
    totalStep++;
    const { system } = getPrompt(
      diaryContext,
      allQuestions,
      allKeywords,
      false,
      false,
      false,
      false,
      false,
      allKnowledge,
      weightedURLs,
      true,
    );

    schema = SchemaGen.getAgentSchema(false, false, true, false, false, question);
    msgWithKnowledge = composeMsgs(messages, allKnowledge, question, finalAnswerPIP);
    
    try {
      const result = await generator.generateObject({
        model: 'agentBeastMode',
        schema,
        system,
        messages: msgWithKnowledge,
        numRetries: 2
      });
      
      // 确保result.object存在且有必要的属性
      if (!result?.object || typeof result.object !== 'object') {
        // 创建一个安全的默认对象
        logWarning('Beast mode generation returned invalid object, using default answer', { result });
        thisStep = {
          action: 'answer',
          think: '由于处理错误，系统提供了默认回答。',
          answer: question || '您的请求' + '已收到，但处理过程中遇到了技术问题。',
          references: [],
          isFinal: true
        } as AnswerAction;
      } else {
        // 确保action属性存在
        const action = result.object.action || 'answer';
        // 确保对应action的属性对象存在
        const actionProps = (result.object[action] || {}) as Record<string, any>;
        
        thisStep = {
          action: action,
          think: result.object.think || '',
          ...actionProps,
          // 确保answer属性存在（如果action是answer）
          ...(action === 'answer' && !actionProps.answer ? { answer: '系统无法生成有效回答。' } : {})
        } as StepAction;
      }
    } catch (error) {
      // 处理异常情况
      logError('Beast mode generation failed:', { error });
      thisStep = {
        action: 'answer',
        think: '处理请求时发生错误。',
        answer: question ? `关于"${question}"的请求处理过程中遇到了技术问题。` : '处理您的请求时遇到了技术问题。',
        references: [],
        isFinal: true
      } as AnswerAction;
    }
    
    // 确保thisStep是一个有效的AnswerAction
    if (thisStep.action === 'answer' && !thisStep.answer) {
      (thisStep as AnswerAction).answer = '无法生成回答内容。';
    }
    
    // 确保isFinal标志被设置
    (thisStep as AnswerAction).isFinal = true;
    context.actionTracker.trackAction({ totalStep, thisStep, gaps });
  }

  const answerStep = thisStep as AnswerAction;

  // 确保answerStep是有效对象且具有必要的属性
  if (!answerStep || typeof answerStep !== 'object') {
    logError('Invalid answerStep object', { thisStep });
    return {
      result: {
        action: 'answer',
        answer: '处理请求时发生错误。',
        think: '',
        references: [],
        isFinal: true,
        mdAnswer: '处理请求时发生错误。'
      } as AnswerAction,
      context,
      visitedURLs: [],
      readURLs: [],
      allURLs: [],
      imageReferences: undefined,
    };
  }

  // 确保answerStep.answer存在
  if (!answerStep.answer) {
    answerStep.answer = '无法生成有效回答。';
  }

  if (trivialQuestion) {
    answerStep.mdAnswer = buildMdFromAnswer(answerStep);
  } else if (!answerStep.isAggregated) {
    try {
      // 添加日志记录，打印出finalizer处理前的答案
      logInfo('Pre-finalizer answer:', { 
        answerLength: answerStep.answer.length,
        answerPreview: answerStep.answer.substring(0, 200) + '...'
      });
      
      // 使用finalizeAnswer处理答案
      let finalizedAnswer = '';
      try {
        finalizedAnswer = await finalizeAnswer(
          answerStep.answer,
          allKnowledge || [],
          context,
          SchemaGen
        );
      } catch (error) {
        logError('Error in finalizeAnswer:', { error });
        finalizedAnswer = answerStep.answer; // 出错时使用原始答案
      }
      
      // 确保finalizedAnswer不为undefined
      if (!finalizedAnswer) finalizedAnswer = answerStep.answer;
      
      // 应用各种修复函数
      answerStep.answer = repairMarkdownFinal(
        convertHtmlTablesToMd(
          fixBadURLMdLinks(
            fixCodeBlockIndentation(
              repairMarkdownFootnotesOuter(finalizedAnswer)
            ),
            allURLs || {})));
            
      // 添加日志记录，打印出finalizer处理后的答案
      logInfo('Post-finalizer answer:', { 
        answerLength: answerStep.answer.length,
        answerPreview: answerStep.answer.substring(0, 200) + '...'
      });

      // 构建引用
      let references: Array<Reference> = [];
      try {
        const buildReferencesResult = await buildReferences(
          answerStep.answer,
          allWebContents || {},
          context,
          SchemaGen,
          80,
          maxRef,
          minRelScore,
          onlyHostnames || []
        );
        
        // 确保返回结果有效
        if (buildReferencesResult && typeof buildReferencesResult === 'object') {
          if (typeof buildReferencesResult.answer === 'string') {
            answerStep.answer = buildReferencesResult.answer;
          }
          references = Array.isArray(buildReferencesResult.references) ? buildReferencesResult.references : [];
        }
      } catch (error) {
        logError('Error in buildReferences:', { error });
        // 出错时使用空引用数组
        references = [];
      }

      answerStep.answer = answerStep.answer || '无法生成有效回答。';
      answerStep.references = references || []; // 确保references不为undefined
      
      // 更新引用
      try {
        await updateReferences(answerStep, allURLs || {});
      } catch (error) {
        logError('Error in updateReferences:', { error });
        // 确保引用数组存在
        answerStep.references = answerStep.references || [];
      }
      
      // 构建markdown答案
      try {
        answerStep.mdAnswer = repairMarkdownFootnotesOuter(buildMdFromAnswer(answerStep));
      } catch (error) {
        logError('Error building markdown answer:', { error });
        answerStep.mdAnswer = answerStep.answer; // 出错时直接使用answer作为mdAnswer
      }

      // 处理图像引用
      if (imageObjects?.length && withImages) {
        try {
          answerStep.imageReferences = await buildImageReferences(answerStep.answer, imageObjects, context, SchemaGen);
          logDebug('Image references built:', { 
            imageReferences: answerStep.imageReferences?.map(i => ({ 
              url: i?.url, 
              score: i?.relevanceScore, 
              answerChunk: i?.answerChunk 
            })) || []
          });
        } catch (error) {
          logError('Error building image references:', { error });
          answerStep.imageReferences = [];
        }
      }
    } catch (error) {
      // 捕获整个处理流程中的任何错误
      logError('Error in answer processing pipeline:', { error });
      // 确保有基本的回答内容
      answerStep.answer = answerStep.answer || '处理回答时发生错误。';
      answerStep.mdAnswer = answerStep.mdAnswer || answerStep.answer;
      answerStep.references = answerStep.references || [];
    }
  } else if (answerStep.isAggregated) {
    try {
      // 确保candidateAnswers是数组
      const validCandidateAnswers = Array.isArray(candidateAnswers) ? candidateAnswers : [];
      answerStep.answer = validCandidateAnswers.join('\n\n'); // await reduceAnswers(candidateAnswers, context, SchemaGen);
      
      try {
        answerStep.mdAnswer = repairMarkdownFootnotesOuter(buildMdFromAnswer(answerStep));
      } catch (error) {
        logError('Error building aggregated markdown answer:', { error });
        answerStep.mdAnswer = answerStep.answer;
      }
      
      if (withImages && answerStep.imageReferences?.length) {
        try {
          const sortedImages = answerStep.imageReferences.sort((a, b) => ((b?.relevanceScore || 0) - (a?.relevanceScore || 0)));
          logDebug('[agent] all sorted image references:', { count: sortedImages?.length || 0 });
          const dedupImages = dedupImagesWithEmbeddings(sortedImages as ImageObject[], []);
          const filteredImages = filterImages(sortedImages, dedupImages);
          logDebug('[agent] filtered images:', { count: filteredImages?.length || 0 });
          answerStep.imageReferences = filteredImages?.slice(0, 10) || []; // limit to 10 images
        } catch (error) {
          logError('Error processing image references in aggregated mode:', { error });
          answerStep.imageReferences = [];
        }
      }
    } catch (error) {
      logError('Error in aggregated answer processing:', { error });
      // 确保有基本的回答内容
      answerStep.answer = answerStep.answer || '处理聚合回答时发生错误。';
      answerStep.mdAnswer = answerStep.mdAnswer || answerStep.answer;
    }
  }

  // 确保最终的answerStep具有所有必要属性
  if (!answerStep.answer) answerStep.answer = '无法生成有效回答。';
  if (!answerStep.mdAnswer) answerStep.mdAnswer = answerStep.answer;
  if (!answerStep.references) answerStep.references = [];

  // max return 300 urls
  const returnedURLs = weightedURLs?.slice(0, numReturnedURLs)?.filter(r => r?.url)?.map(r => r.url) || [];
  
  // 确保thisStep是有效对象
  if (!thisStep || typeof thisStep !== 'object') {
    logError('Invalid thisStep object at return point', { thisStep });
    thisStep = {
      action: 'answer',
      answer: '处理请求时发生错误。',
      think: '',
      references: [],
      isFinal: true,
      mdAnswer: '处理请求时发生错误。'
    } as AnswerAction;
  }
  
  // 确保context是有效对象
  let safeContext = context;
  if (!safeContext || typeof safeContext !== 'object') {
    logError('Invalid context object at return point', { context });
    safeContext = {
      tokenTracker: new TokenTracker(),
      actionTracker: new ActionTracker()
    };
  }
  
  return {
    result: thisStep,
    context: safeContext,
    visitedURLs: returnedURLs, // deprecated
    readURLs: visitedURLs?.filter(url => !badURLs.includes(url)) || [],
    allURLs: weightedURLs?.map(r => r?.url) || [],
    imageReferences: withImages ? ((thisStep as AnswerAction).imageReferences || undefined) : undefined,
  };
}

async function storeContext(prompt: string, schema: any, memory: {
  allContext: StepAction[];
  allKeywords: string[];
  allQuestions: string[];
  allKnowledge: KnowledgeItem[];
  weightedURLs: BoostedSearchSnippet[];
  msgWithKnowledge: CoreMessage[];
}
  , step: number) {

  const { allContext, allKeywords, allQuestions, allKnowledge, weightedURLs, msgWithKnowledge } = memory;
  if ((process as any).asyncLocalContext?.available?.()) {

    (process as any).asyncLocalContext.ctx.promptContext = {
      prompt,
      schema,
      allContext,
      allKeywords,
      allQuestions,
      allKnowledge,
      step
    };
    return;
  }

  try {
    await fs.writeFile(`prompt-${step}.txt`, `
Prompt:
${prompt}

JSONSchema:
${JSON.stringify(zodToJsonSchema(schema), null, 2)}
`);
    await fs.writeFile('context.json', JSON.stringify(allContext, null, 2));
    await fs.writeFile('queries.json', JSON.stringify(allKeywords, null, 2));
    await fs.writeFile('questions.json', JSON.stringify(allQuestions, null, 2));
    await fs.writeFile('knowledge.json', JSON.stringify(allKnowledge, null, 2));
    await fs.writeFile('urls.json', JSON.stringify(weightedURLs, null, 2));
    await fs.writeFile('messages.json', JSON.stringify(msgWithKnowledge, null, 2));
  } catch (error) {
    logError('Context storage failed:', {
      error: error instanceof Error ? error.message : String(error)
    });
  }
}

export async function main() {
  const question = process.argv[2] || "";
  const {
    result: finalStep,
    context: tracker,
    visitedURLs: visitedURLs
  } = await getResponse(question) as { result: AnswerAction; context: TrackerContext; visitedURLs: string[] };
  logInfo('Final Answer:', { answer: finalStep.answer });
  logInfo('Visited URLs:', { urls: visitedURLs });

  tracker.tokenTracker.printSummary();
}

if (require.main === module) {
  main().catch(error => {
    logError('Main execution error:', {
      error: error instanceof Error ? error.message : String(error)
    });
  });
}