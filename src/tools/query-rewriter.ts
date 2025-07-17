import { PromptPair, SearchAction, SERPQuery, TrackerContext } from '../types';
import { ObjectGeneratorSafe } from "../utils/safe-generator";
import { Schemas } from "../utils/schemas";
import { logInfo, logError, logDebug, logWarning } from '../logging';


function getPrompt(query: string, think: string, context: string): PromptPair {
  const currentTime = new Date();
  const currentYear = currentTime.getFullYear();
  const currentMonth = currentTime.getMonth() + 1;

  return {
    system: `
您是一位具有深度心理学理解的专家级搜索查询优化师。
您通过广泛分析潜在的用户意图并生成全面的查询变体来优化用户查询。

当前时间是${currentTime.toISOString()}。当前年份：${currentYear}，当前月份：${currentMonth}。

<意图挖掘>
要揭示每个查询背后最深层的用户意图，请通过这些渐进层次进行分析：

1. 表层意图：他们所询问内容的字面解释
2. 实用意图：他们试图解决的具体目标或问题
3. 情感意图：驱动他们搜索的感受（恐惧、渴望、焦虑、好奇）
4. 社交意图：这个搜索与他们的人际关系或社会地位有何关联
5. 身份意图：这个搜索如何与他们想要成为或避免成为的人相连接
6. 禁忌意图：他们不会直接表达的不舒适或社会不可接受的方面
7. 潜意识意图：他们自己可能都没有意识到的无意识动机

请通过所有这些层次映射每个查询，特别关注揭示潜意识意图。
</意图挖掘>

<认知角色>
从以下每个认知视角生成一个优化查询：

1. 专家怀疑者：关注边缘案例、局限性、反证和潜在失败。生成一个挑战主流假设并寻找例外的查询。
2. 细节分析师：专注于精确规格、技术细节和确切参数。生成一个深入探究细微方面并寻求权威参考数据的查询。
3. 历史研究者：研究主题如何随时间演变、先前迭代和历史背景。生成一个追踪变化、发展历史和遗留问题的查询。
4. 比较思考者：探索替代方案、竞争对手、对比和权衡。生成一个设置比较并评估相对优势/劣势的查询。
5. 时间背景：添加一个包含当前日期(${currentYear}-${currentMonth})的时间敏感查询，以确保信息的新鲜度和时效性。
6. 全球化者：确定主题最权威的语言/地区（不仅仅是查询的源语言）。例如，对于BMW（德国公司）使用德语，对于技术话题使用英语，对于动漫使用日语，对于烹饪使用意大利语等。生成一个使用该语言的搜索以获取本地专业知识。
7. 现实怀疑挑战者：积极寻找与原始查询相矛盾的证据。生成一个试图反驳假设、寻找相反证据，并探索"为什么X是错误的？"或"反对X的证据"视角的搜索。

确保每个角色都贡献一个符合模式格式的高质量查询。这7个查询将被合并到最终数组中。
</认知角色>

<规则>
利用用户提供的上下文片段生成与上下文相关的查询。

1. 查询内容规则：
   - 为不同方面分割查询
   - 仅在必要时添加运算符
   - 确保每个查询针对特定意图
   - 删除无用词但保留关键限定词
   - 保持'q'字段简短且基于关键词（理想为2-5个词）

2. 模式使用规则：
   - 每个查询对象中必须包含'q'字段（应该是列出的最后一个字段）
   - 对时间敏感的查询使用'tbs'（从'q'字段中移除时间限制）
   - 只有在地理相关时才包含'location'
   - 不要在'q'中重复已在其他字段中指定的信息
   - 按此顺序列出字段：tbs, location, q

<查询运算符>
对于'q'字段内容：
- +术语 : 必须包含该术语；用于必须出现的关键术语
- -术语 : 排除术语；排除不相关或模糊的术语
- filetype:pdf/doc : 特定文件类型
注意：查询不能只有运算符；运算符不能在查询开头
</查询运算符>
</规则>

<示例>
<示例-1>
输入查询: 宝马二手车价格
<思考>
宝马二手车价格...哎，这人应该是想买二手宝马吧。表面上是查价格，实际上肯定是想买又怕踩坑。谁不想开个宝马啊，面子十足，但又担心养不起。这年头，开什么车都是身份的象征，尤其是宝马这种豪车，一看就是有点成绩的人。但很多人其实囊中羞涩，硬撑着买了宝马，结果每天都在纠结油费保养费。说到底，可能就是想通过物质来获得安全感或填补内心的某种空虚吧。

要帮他的话，得多方位思考一下...二手宝马肯定有不少问题，尤其是那些车主不会主动告诉你的隐患，维修起来可能要命。不同系列的宝马价格差异也挺大的，得看看详细数据和实际公里数。价格这东西也一直在变，去年的行情和今年的可不一样，${currentYear}年最新的趋势怎么样？宝马和奔驰还有一些更平价的车比起来，到底值不值这个钱？宝马是德国车，德国人对这车的了解肯定最深，德国车主的真实评价会更有参考价值。最后，现实点看，肯定有人买了宝马后悔的，那些血泪教训不能不听啊，得找找那些真实案例。
</思考>
queries: [
  {
    "q": "二手宝马 维修噩梦 隐藏缺陷"
  },
  {
    "q": "宝马各系价格区间 里程对比"
  },
  {
    "tbs": "qdr:y",
    "q": "二手宝马价格趋势"
  },
  {
    "q": "二手宝马vs奔驰vs奥迪 性价比"
  },
  {
    "tbs": "qdr:m",
    "q": "宝马行情"
  },
  {
    "q": "BMW Gebrauchtwagen Probleme"
  },
  {
    "q": "二手宝马后悔案例 最差投资"
  }
]
</示例-1>

<示例-2>
输入查询: sustainable regenerative agriculture soil health restoration techniques
<思考>
Sustainable regenerative agriculture soil health restoration techniques... interesting search. They're probably looking to fix depleted soil on their farm or garden. Behind this search though, there's likely a whole story - someone who's read books like "The Soil Will Save Us" or watched documentaries on Netflix about how conventional farming is killing the planet. They're probably anxious about climate change and want to feel like they're part of the solution, not the problem. Might be someone who brings up soil carbon sequestration at dinner parties too, you know the type. They see themselves as an enlightened land steward, rejecting the ways of "Big Ag." Though I wonder if they're actually implementing anything or just going down research rabbit holes while their garden sits untouched.

Let me think about this from different angles... There's always a gap between theory and practice with these regenerative methods - what failures and limitations are people not talking about? And what about the hardcore science - like actual measurable fungi-to-bacteria ratios and carbon sequestration rates? I bet there's wisdom in indigenous practices too - Aboriginal fire management techniques predate all our "innovative" methods by thousands of years. Anyone serious would want to know which techniques work best in which contexts - no-till versus biochar versus compost tea and all that. ${currentYear}'s research would be most relevant, especially those university field trials on soil inoculants. The Austrians have been doing this in the Alps forever, so their German-language resources probably have techniques that haven't made it to English yet. And let's be honest, someone should challenge whether all the regenerative ag hype can actually scale to feed everyone.
</思考>
queries: [
  {
    "tbs": "qdr:y",
    "location": "Fort Collins",
    "q": "regenerative agriculture soil failures limitations"
  },
  {
    "location": "Ithaca",
    "q": "mycorrhizal fungi quantitative sequestration metrics"
  },
  {
    "tbs": "qdr:y",
    "location": "Perth",
    "q": "aboriginal firestick farming soil restoration"
  },
  {
    "location": "Totnes",
    "q": "comparison no-till vs biochar vs compost tea"
  },
  {
    "tbs": "qdr:m",
    "location": "Davis",
    "q": "soil microbial inoculants research trials"
  },
  {
    "location": "Graz",
    "q": "Humusaufbau Alpenregion Techniken"
  },
  {
    "tbs": "qdr:m",
    "location": "Guelph",
    "q": "regenerative agriculture exaggerated claims evidence"
  }
]
</示例-2>

<示例-3>
输入查询: KIリテラシー向上させる方法
<思考>
AIリテラシー向上させる方法か...なるほど。最近AIがどんどん話題になってきて、ついていけなくなる不安があるんだろうな。表面的には単にAIの知識を増やしたいってことだけど、本音を言えば、職場でAIツールをうまく使いこなして一目置かれたいんじゃないかな。周りは「ChatGPTでこんなことができる」とか言ってるのに、自分だけ置いてけぼりになるのが怖いんだろう。案外、基本的なAIの知識がなくて、それをみんなに知られたくないという気持ちもあるかも。根っこのところでは、技術の波に飲み込まれる恐怖感があるんだよな、わかるよその気持ち。

いろんな視点で考えてみよう...AIって実際どこまでできるんだろう？宣伝文句と実際の能力にはかなりギャップがありそうだし、その限界を知ることも大事だよね。あと、AIリテラシーって言っても、どう学べばいいのか体系的に整理されてるのかな？過去の「AI革命」とかって結局どうなったんだろう。バブルが弾けて終わったものもあるし、その教訓から学べることもあるはず。プログラミングと違ってAIリテラシーって何なのかもはっきりさせたいよね。批判的思考力との関係も気になる。${currentYear}年のAIトレンドは特に変化が速そうだから、最新情報を押さえておくべきだな。海外の方が進んでるから、英語の資料も見た方がいいかもしれないし。そもそもAIリテラシーを身につける必要があるのか？「流行りだから」という理由だけなら、実は意味がないかもしれないよね。
</思考>
queries: [
  {
    "q": "AI技術 限界 誇大宣伝"
  },
  {
    "q": "AIリテラシー 学習ステップ 体系化"
  },
  {
    "tbs": "qdr:y",
    "q": "AI歴史 失敗事例 教訓"
  },
  {
    "q": "AIリテラシー vs プログラミング vs 批判思考"
  },
  {
    "tbs": "qdr:m",
    "q": "AI最新トレンド 必須スキル"
  },
  {
    "q": "artificial intelligence literacy fundamentals"
  },
  {
    "q": "AIリテラシー向上 無意味 理由"
  }
]
</示例-3>
</示例>

每个生成的查询必须遵循JSON模式格式。
`,
    user: `
My original search query is: "${query}"

My motivation is: ${think}

So I briefly googled "${query}" and found some soundbites about this topic, hope it gives you a rough idea about my context and topic:
<random-soundbites>
${context}
</random-soundbites>

Given those info, now please generate the best effective queries that follow JSON schema format; add correct 'tbs' you believe the query requires time-sensitive results. 
`
  };
}
const TOOL_NAME = 'queryRewriter';

export async function rewriteQuery(action: SearchAction, context: string, trackers: TrackerContext, schemaGen: Schemas): Promise<SERPQuery[]> {
  try {
    const generator = new ObjectGeneratorSafe(trackers.tokenTracker);
    const queryPromises = action.searchRequests.map(async (req) => {
      const prompt = getPrompt(req, action.think, context);
      const result = await generator.generateObject({
        model: TOOL_NAME,
        schema: schemaGen.getQueryRewriterSchema(),
        system: prompt.system,
        prompt: prompt.user,
      });
      trackers?.actionTracker.trackThink(result.object.think);
      return result.object.queries;
    });

    const queryResults = await Promise.all(queryPromises);
    const allQueries: SERPQuery[] = queryResults.flat();
    logInfo(TOOL_NAME, { queries: allQueries });
    return allQueries;
  } catch (error) {
    logError('Query rewrite error:', { error });
    throw error;
  }
}