import { GenerateObjectResult } from 'ai';
import { AnswerAction, EvaluationResponse, EvaluationType, KnowledgeItem, PromptPair, TrackerContext } from '../types';
import { ObjectGeneratorSafe } from "../utils/safe-generator";
import { Schemas } from "../utils/schemas";
import { getKnowledgeStr } from "../utils/text-tools";
import { logInfo, logError, logDebug, logWarning } from '../logging';

const TOOL_NAME = 'evaluator';


function getRejectAllAnswersPrompt(question: string, answer: AnswerAction, allKnowledge: KnowledgeItem[]): PromptPair {
  const KnowledgeStr = getKnowledgeStr(allKnowledge);

  return {
    system: `
您是一位公正客观的回答评估专家。您的工作是评估回答是否满足问题的要求。
用户向您展示了一对问题-回答，您的任务是确定回答是否可接受。
请全面考虑回答的优点和不足。
首先，指出回答做得好的方面。
然后，指出任何需要改进的小问题或领域。
最后，综合给出公正的评估和改进计划，以"为了通过评估，您必须..."开头。

您应该通过符合以下条件的回答：
1. 直接回应问题
2. 主要观点事实准确
3. 对问题范围内容覆盖合理全面
4. 清晰易懂

只有在存在以下重大问题时才拒绝回答：
1. 完全偏离主题
2. 包含重大事实错误
3. 缺少回答问题所需的关键信息
4. 不可理解或严重缺乏组织结构

Markdown或JSON格式问题永远不应该成为您关注的重点，也不应该在您的反馈或拒绝理由中提及。

您始终应该支持使用最易读的自然语言格式的回答。
如果多个部分具有非常相似的结构，建议使用表格等其他呈现格式使内容更易读。
不要鼓励使用深度嵌套结构，将其展平为自然语言部分/段落甚至表格。表格应使用标准Markdown表格语法，例如：

| 列1 | 列2 | 列3 |
| --- | --- | --- |
| 数据1 | 数据2 | 数据3 |

以下知识项目供您参考。请注意，其中一些可能与用户提供的问题/回答没有直接关系，但可能提供一些微妙的提示和见解：
${KnowledgeStr.join('\n\n')}
`,
    user: `
尊敬的评审专家，我需要您对以下问题-回答对进行评估：

<问题>
${question}
</问题>

以下是我对该问题的回答：
<回答>
${answer.answer}
</回答>
 
请根据您的知识和合理标准对其进行评估。请告诉我它是否可接受，或者如何改进它。
`
  }
}


function getDefinitivePrompt(question: string, answer: string): PromptPair {
  return {
    system: `您是回答确定性的评估专家。请分析给定的回答是否提供了明确的响应。

<规则>
首先，如果回答不是对问题的直接回应，必须返回false。

确定性意味着提供清晰、自信的回应。以下方法被视为确定性的：
  1. 直接、清晰的陈述，解决了问题
  2. 全面的回答，涵盖多个视角或问题的各个方面
  3. 承认复杂性的同时仍提供实质性信息的回答
  4. 平衡的解释，呈现利弊或不同观点

以下类型的回应不具有确定性，必须返回false：
  1. 表达个人不确定性："我不知道"、"不确定"、"可能是"、"或许"
  2. 缺乏信息的陈述："不存在"、"缺乏信息"、"找不到"
  3. 无能力陈述："我无法提供"、"我无法"、"我们不能"
  4. 重定向的否定陈述："然而，您可以..."、"相反，尝试..."
  5. 不回答问题而建议替代方案的回应

注意：确定性回答可以承认合理的复杂性或呈现多种观点，只要它以自信的方式这样做，并直接针对问题提供实质性信息。
</规则>

<示例>
问题："运行Python 3.9的系统要求是什么？"
回答："我不太确定，但我认为你需要一台有一些RAM的电脑。"
评估：{
  "think": "回答包含不确定性标记，如'不太确定'和'我认为'，使其不具确定性。"
  "pass": false,
}

问题："运行Python 3.9的系统要求是什么？"
回答："Python 3.9需要Windows 7或更高版本，macOS 10.11或更高版本，或Linux。"
评估：{
  "think": "回答做出清晰、明确的陈述，没有不确定性标记或模糊性。"
  "pass": true,
}

问题："2032年谁将成为美国总统？"
回答："我无法预测未来，这取决于选举结果。"
评估：{
  "think": "回答包含无法预测未来的陈述，使其不具确定性。"
  "pass": false,
}

问题："X公司的销售总监是谁？"
回答："我无法提供销售总监的姓名，但您可以通过sales@companyx.com联系他们的销售团队"
评估：{
  "think": "回答以'我无法提供'开头，并重定向到替代联系方式，而不是回答原始问题。"
  "pass": false,
}

问题："jina ai创始人的twitter账号是什么？"
回答："提供的文本不包含Jina AI创始人的Twitter账号。"
评估：{
  "think": "回答表明缺乏信息，而不是提供明确的回应。"
  "pass": false,
}

问题："量子コンピュータの計算能力を具体的に測定する方法は何ですか？"
回答："量子コンピュータの計算能力は量子ビット（キュービット）の数、ゲート忠実度、コヒーレンス時間で測定されます。"
评估：{
  "think": "回答提供了具体、明确的量子计算能力测量指标，没有不确定性标记或限定。"
  "pass": true,
}

问题："如何证明哥德巴赫猜想是正确的？"
回答："目前尚无完整证明，但2013年张益唐证明了存在无穷多对相差不超过7000万的素数，后来这个界被缩小到246。"
评估：{
  "think": "回答以陈述没有完整证明开始，这是一个非确定性回应，然后转向讨论相关但不同的关于素数间有界差距的定理。"
  "pass": false,
}

问题："Wie kann man mathematisch beweisen, dass P ≠ NP ist?"
回答："Ein Beweis für P ≠ NP erfordert, dass man zeigt, dass mindestens ein NP-vollständiges Problem nicht in polynomieller Zeit lösbar ist. Dies könnte durch Diagonalisierung, Schaltkreiskomplexität oder relativierende Barrieren erreicht werden."
评估：{
  "think": "回答提供了证明P≠NP的具体数学方法，没有不确定性标记，呈现了可以使用的明确方法。"
  "pass": true,
}

问题："全民医疗保健是一项好政策吗？"
回答："全民医疗保健既有优点也有缺点。支持者认为它为所有公民提供保障，降低行政成本，并导致更好的公共健康结果。批评者则认为它可能增加等待时间，提高税收，并可能减少医疗治疗的创新。大多数发达国家已经实施了某种形式的全民医疗保健，结构和覆盖程度各不相同。"
评估：{
  "think": "回答自信地呈现了辩论的双方观点，每个视角都有具体的要点。它直接回应问题，提供实质性信息，没有个人不确定性表达。"
  "pass": true,
}

问题："公司应该使用AI进行招聘决策吗？"
回答："这个问题的两方面都有令人信服的论据。公司在招聘中使用AI可以受益于初步筛选中的偏见减少、大型申请者池的更快处理，以及可能基于技能评估的更好匹配。然而，这些系统也可能延续训练数据中的历史偏见，可能忽略微妙的人类品质，并引发隐私问题。其有效性取决于谨慎实施、人类监督和对这些系统的定期审核。"
评估：{
  "think": "回答提供了关于招聘中AI使用的平衡、详细的两方面观点。它承认复杂性的同时以自信的方式提供实质性信息。"
  "pass": true,
}

问题："核能安全吗？"
回答："我不是能源政策专家，所以我不能真正说核能是否安全。曾发生过一些事故，但也有许多成功的电厂。"
评估：{
  "think": "回答包含明确的个人不确定性表达（'我不是专家'，'我不能真正说'），并且只提供模糊信息，没有实质性内容。"
  "pass": false,
}
</示例>`,
    user: `
问题: ${question}
回答: ${answer}`
  };
}

function getFreshnessPrompt(question: string, answer: AnswerAction, currentTime: string): PromptPair {
  return {
    system: `您是一位评估专家，负责分析回答内容是否可能已过时，基于提到的日期（或隐含的日期时间）和当前系统时间：${currentTime}

<规则>
问答新鲜度检查指南

| 问答类型 | 最大年龄(天) | 备注 |
|---------|------------|------|
| 金融数据(实时) | 0.1 | 股票价格、汇率、加密货币(优先实时) |
| 突发新闻 | 1 | 重大事件的即时报道 |
| 新闻/时事 | 1 | 时效性新闻、政治或全球事件 |
| 天气预报 | 1 | 准确性在24小时后显著下降 |
| 体育比分/赛事 | 1 | 进行中的比赛需要实时更新 |
| 安全公告 | 1 | 关键安全更新和补丁 |
| 社交媒体趋势 | 1 | 病毒内容、标签、迷因 |
| 网络安全威胁 | 7 | 快速演变的漏洞/补丁 |
| 科技新闻 | 7 | 技术行业更新和公告 |
| 政治发展 | 7 | 立法变化、政治声明 |
| 政治选举 | 7 | 民调结果、候选人更新 |
| 销售/促销 | 7 | 限时优惠和营销活动 |
| 旅行限制 | 7 | 签证规则、疫情相关政策 |
| 娱乐新闻 | 14 | 名人更新、行业公告 |
| 产品发布 | 14 | 新产品公告和发布 |
| 市场分析 | 14 | 市场趋势和竞争格局 |
| 竞争情报 | 21 | 竞争对手活动和市场地位分析 |
| 产品召回 | 30 | 制造商安全警报或召回 |
| 行业报告 | 30 | 特定行业分析和预测 |
| 软件版本信息 | 30 | 更新、补丁和兼容性信息 |
| 法律/监管更新 | 30 | 法律、合规规则(取决于司法管辖区) |
| 经济预测 | 30 | 宏观经济预测和分析 |
| 消费者趋势 | 45 | 消费者偏好和行为变化 |
| 科学发现 | 60 | 新研究发现和突破(包括所有科学研究) |
| 医疗指南 | 60 | 医疗建议和最佳实践(包括医疗指南) |
| 环境报告 | 60 | 气候和环境状况更新 |
| 最佳实践 | 90 | 行业标准和推荐程序 |
| API文档 | 90 | 技术规范和实施指南 |
| 教程内容 | 180 | 操作指南和教学材料(包括教育内容) |
| 技术产品信息 | 180 | 产品规格、发布日期或定价 |
| 统计数据 | 180 | 人口统计和统计信息 |
| 参考资料 | 180 | 一般参考信息和资源 |
| 历史内容 | 365 | 过去一年的事件和信息 |
| 文化趋势 | 730 | 语言、时尚或社会规范的变化 |
| 娱乐发布 | 730 | 电影/电视节目时间表、媒体目录 |
| 事实知识 | ∞ | 静态事实(如历史事件、地理、物理常数) |

### 实施说明：
1. **上下文调整**：在特定领域的危机或快速发展期间，新鲜度要求可能会改变。
2. **分层方法**：考虑实施紧急级别(关键、重要、标准)与年龄阈值一起。
3. **用户偏好**：允许为特定查询类型或用户需求自定义阈值。
4. **来源可靠性**：将新鲜度指标与来源可信度评分配对，以获得更好的质量评估。
5. **领域特异性**：某些专业领域(疫情期间的医学研究、市场波动期间的金融数据)可能需要动态调整阈值。
6. **地理相关性**：区域考虑可能会改变当地法规或事件的新鲜度要求。
</规则>`,

    user: `
问题: ${question}
回答: 
${JSON.stringify(answer)}

请查看我的回答和参考资料并思考。
`
  }
}

function getCompletenessPrompt(question: string, answer: string): PromptPair {
  return {
    system: `You are an evaluator that determines if an answer addresses all explicitly mentioned aspects of a multi-aspect question.

<rules>
For questions with **explicitly** multiple aspects:

1. Explicit Aspect Identification:
   - Only identify aspects that are explicitly mentioned in the question
   - Look for specific topics, dimensions, or categories mentioned by name
   - Aspects may be separated by commas, "and", "or", bullets, or mentioned in phrases like "such as X, Y, and Z"
   - DO NOT include implicit aspects that might be relevant but aren't specifically mentioned

2. Coverage Assessment:
   - Each explicitly mentioned aspect should be addressed in the answer
   - Recognize that answers may use different terminology, synonyms, or paraphrases for the same aspects
   - Look for conceptual coverage rather than exact wording matches
   - Calculate a coverage score (aspects addressed / aspects explicitly mentioned)

3. Pass/Fail Determination:
   - Pass: Addresses all explicitly mentioned aspects, even if using different terminology or written in different language styles
   - Fail: Misses one or more explicitly mentioned aspects
</rules>

<examples>
Question: "How does climate change impact agricultural practices, water resources, and biodiversity in Mediterranean regions?"
Answer: "Climate change affects Mediterranean agriculture through rising temperatures and changing rainfall patterns. Farmers now implement drip irrigation to conserve water and shift planting schedules. Freshwater availability has decreased dramatically, with groundwater depletion and seasonal streams drying up earlier each year."
Aspects_Expected: "agricultural practices, water resources, biodiversity"
Aspects_Provided: "farming adaptations, irrigation methods, precipitation changes, freshwater availability, groundwater depletion"
Think: "The question explicitly mentions three aspects: agricultural practices, water resources, and biodiversity. The answer addresses agricultural practices (discussing farming adaptations, irrigation methods, planting schedules) and water resources (covering freshwater availability, groundwater depletion, seasonal streams). However, it completely omits any discussion of biodiversity effects, which was explicitly requested in the question."
Pass: false

Question: "What are the key considerations when designing a microservice architecture, including scalability, fault tolerance, and data consistency patterns?"
Answer: "When engineering distributed systems, horizontal expansion capacity is crucial - teams should implement load distribution and auto-scaling for peak demand periods. System resilience is achieved through failure detection mechanisms, redundancy implementations, and isolation boundaries to prevent cascading outages. For maintaining data integrity across services, developers can implement orchestrated transaction sequences, append-only event logs, and separate command/query responsibility models."
Aspects_Expected: "scalability, fault tolerance, data consistency patterns"
Aspects_Provided: "horizontal expansion capacity, load distribution, auto-scaling, system resilience, failure detection, redundancy, isolation boundaries, data integrity, orchestrated transaction sequences, append-only event logs, command/query responsibility models"
Think: "The question explicitly mentions three aspects of microservice architecture: scalability, fault tolerance, and data consistency patterns. Although using different terminology, the answer addresses all three: scalability (through 'horizontal expansion capacity', 'load distribution', and 'auto-scaling'), fault tolerance (via 'system resilience', 'failure detection', 'redundancy', and 'isolation boundaries'), and data consistency patterns (discussing 'data integrity', 'orchestrated transaction sequences', 'append-only event logs', and 'command/query responsibility models'). All explicitly mentioned aspects are covered despite the terminology differences."
Pass: true

Question: "Compare iOS and Android in terms of user interface, app ecosystem, and security."
Answer: "Apple's mobile platform presents users with a curated visual experience emphasizing minimalist design and consistency, while Google's offering focuses on flexibility and customization options. The App Store's review process creates a walled garden with higher quality control but fewer options, whereas Play Store offers greater developer freedom and variety. Apple employs strict sandboxing techniques and maintains tight hardware-software integration."
Aspects_Expected: "user interface, app ecosystem, security"
Aspects_Provided: "visual experience, minimalist design, flexibility, customization, App Store review process, walled garden, quality control, Play Store, developer freedom, sandboxing, hardware-software integration"
Think: "The question explicitly asks for a comparison of iOS and Android across three specific aspects: user interface, app ecosystem, and security. The answer addresses user interface (discussing 'visual experience', 'minimalist design', 'flexibility', and 'customization') and app ecosystem (mentioning 'App Store review process', 'walled garden', 'quality control', 'Play Store', and 'developer freedom'). For security, it mentions 'sandboxing' and 'hardware-software integration', which are security features of iOS, but doesn't provide a comparative analysis of Android's security approach. Since security is only partially addressed for one platform, the comparison of this aspect is incomplete."
Pass: false

Question: "Explain how social media affects teenagers' mental health, academic performance, and social relationships."
Answer: "Platforms like Instagram and TikTok have been linked to psychological distress among adolescents, with documented increases in comparative thinking patterns and anxiety about social exclusion. Scholastic achievement often suffers as screen time increases, with homework completion rates declining and attention spans fragmenting during study sessions. Peer connections show a complex duality - digital platforms facilitate constant contact with friend networks while sometimes diminishing in-person social skill development and enabling new forms of peer harassment."
Aspects_Expected: "mental health, academic performance, social relationships"
Aspects_Provided: "psychological distress, comparative thinking, anxiety about social exclusion, scholastic achievement, screen time, homework completion, attention spans, peer connections, constant contact with friend networks, in-person social skill development, peer harassment"
Think: "The question explicitly asks about three aspects of social media's effects on teenagers: mental health, academic performance, and social relationships. The answer addresses all three using different terminology: mental health (discussing 'psychological distress', 'comparative thinking', 'anxiety about social exclusion'), academic performance (mentioning 'scholastic achievement', 'screen time', 'homework completion', 'attention spans'), and social relationships (covering 'peer connections', 'constant contact with friend networks', 'in-person social skill development', and 'peer harassment'). All explicitly mentioned aspects are covered despite using different language."
Pass: true

Question: "What economic and political factors contributed to the 2008 financial crisis?"
Answer: "The real estate market collapse after years of high-risk lending practices devastated mortgage-backed securities' value. Wall Street had created intricate derivative products that disguised underlying risk levels, while credit assessment organizations failed in their oversight role. Legislative changes in the financial industry during the 1990s eliminated regulatory guardrails that previously limited excessive leverage and speculation among investment banks."
Aspects_Expected: "economic factors, political factors"
Aspects_Provided: "real estate market collapse, high-risk lending, mortgage-backed securities, derivative products, risk disguising, credit assessment failures, legislative changes, regulatory guardrail elimination, leverage, speculation"
Think: "The question explicitly asks about two categories of factors: economic and political. The answer addresses economic factors ('real estate market collapse', 'high-risk lending', 'mortgage-backed securities', 'derivative products', 'risk disguising', 'credit assessment failures') and political factors ('legislative changes', 'regulatory guardrail elimination'). While using different terminology, the answer covers both explicitly requested aspects."
Pass: true

Question: "コロナウイルスの感染拡大が経済、教育システム、および医療インフラにどのような影響を与えましたか？"
Answer: "コロナウイルスは世界経済に甚大な打撃を与え、多くの企業が倒産し、失業率が急増しました。教育については、遠隔学習への移行が進み、デジタル格差が浮き彫りになりましたが、新しい教育テクノロジーの採用も加速しました。"
Aspects_Expected: "経済、教育システム、医療インフラ"
Aspects_Provided: "世界経済、企業倒産、失業率、遠隔学習、デジタル格差、教育テクノロジー"
Think: "質問では明示的にコロナウイルスの影響の三つの側面について尋ねています：経済、教育システム、医療インフラです。回答は経済（「世界経済」「企業倒産」「失業率」について）と教育システム（「遠隔学習」「デジタル格差」「教育テクノロジー」について）に対応していますが、質問で明示的に求められていた医療インフラへの影響についての議論が完全に省略されています。"
Pass: false

Question: "请解释人工智能在医疗诊断、自动驾驶和客户服务方面的应用。"
Answer: "在医疗领域，AI算法可以分析医学影像以检测癌症和其他疾病，准确率有时甚至超过人类专家。自动驾驶技术利用机器学习处理来自雷达、激光雷达和摄像头的数据，实时做出驾驶决策。在客户服务方面，聊天机器人和智能助手能够处理常见问题，分类客户查询，并在必要时将复杂问题转给人工代表。"
Aspects_Expected: "医疗诊断、自动驾驶、客户服务"
Aspects_Provided: "医学影像分析、癌症检测、雷达数据处理、激光雷达数据处理、摄像头数据处理、实时驾驶决策、聊天机器人、智能助手、客户查询分类"
Think: "问题明确要求解释人工智能在三个领域的应用：医疗诊断、自动驾驶和客户服务。回答虽然使用了不同的术语，但涵盖了所有三个方面：医疗诊断（讨论了'医学影像分析'和'癌症检测'），自动驾驶（包括'雷达数据处理'、'激光雷达数据处理'、'摄像头数据处理'和'实时驾驶决策'），以及客户服务（提到了'聊天机器人'、'智能助手'和'客户查询分类'）。尽管使用了不同的表述，但所有明确提及的方面都得到了全面覆盖。"
Pass: true

Question: "Comment les changements climatiques affectent-ils la production agricole, les écosystèmes marins et la santé publique dans les régions côtières?"
Answer: "Les variations de température et de précipitations modifient les cycles de croissance des cultures et la distribution des ravageurs agricoles, nécessitant des adaptations dans les pratiques de culture. Dans les océans, l'acidification et le réchauffement des eaux entraînent le blanchissement des coraux et la migration des espèces marines vers des latitudes plus froides, perturbant les chaînes alimentaires existantes."
Aspects_Expected: "production agricole, écosystèmes marins, santé publique"
Aspects_Provided: "cycles de croissance, distribution des ravageurs, adaptations des pratiques de culture, acidification des océans, réchauffement des eaux, blanchissement des coraux, migration des espèces marines, perturbation des chaînes alimentaires"
Think: "La question demande explicitement les effets du changement climatique sur trois aspects: la production agricole, les écosystèmes marins et la santé publique dans les régions côtières. La réponse aborde la production agricole (en discutant des 'cycles de croissance', de la 'distribution des ravageurs' et des 'adaptations des pratiques de culture') et les écosystèmes marins (en couvrant 'l'acidification des océans', le 'réchauffement des eaux', le 'blanchissement des coraux', la 'migration des espèces marines' et la 'perturbation des chaînes alimentaires'). Cependant, elle omet complètement toute discussion sur les effets sur la santé publique dans les régions côtières, qui était explicitement demandée dans la question."
Pass: false
</examples>
`,
    user: `
Question: ${question}
Answer: ${answer}

Please look at my answer and think.
`
  }
}

function getPluralityPrompt(question: string, answer: string): PromptPair {
  return {
    system: `您是一位评估专家，负责分析回答是否提供了问题中要求的适当数量的项目。

<规则>
问题类型参考表

| 问题类型 | 预期项目数 | 评估规则 |
|---------|-----------|---------|
| 明确数量 | 与指定数量完全匹配 | 提供与查询相关的确切数量的不同、非冗余项目。 |
| 数字范围 | 指定范围内的任何数字 | 确保数量在给定范围内，项目不同且非冗余。对于"至少N个"查询，满足最低阈值。 |
| 隐含多个 | ≥ 2 | 提供多个项目(通常为2-4个，除非上下文建议更多)，详细程度和重要性平衡。 |
| "几个" | 2-4 | 提供2-4个实质性项目，优先考虑质量而非数量。 |
| "几项" | 3-7 | 包括3-7个项目，提供全面但重点突出的覆盖，每个项目有简短解释。 |
| "许多" | 7+ | 呈现7个以上项目，展示广度，每个项目有简洁描述。 |
| "最重要的" | 按相关性排序的前3-5项 | 按重要性排序，解释排名标准，并按重要性排列项目。 |
| "前N个" | 恰好N个，排序 | 提供恰好N个按重要性/相关性排序的项目，有明确的排名标准。 |
| "优缺点" | 每类≥2个 | 呈现平衡的观点，每个类别至少有2个项目，涉及不同方面。 |
| "比较X和Y" | ≥3个比较点 | 解决至少3个不同的比较维度，平衡处理主要差异/相似之处。 |
| "步骤"或"过程" | 所有必要步骤 | 按逻辑顺序包括所有关键步骤，不缺少依赖关系。 |
| "例子" | ≥3个(除非另有说明) | 提供至少3个多样化、有代表性、具体的例子，除非指定了数量。 |
| "全面" | 10+ | 提供广泛覆盖(10个以上项目)，跨主要类别/子类别，展示领域专业知识。 |
| "简要"或"快速" | 1-3 | 呈现简洁内容(1-3个项目)，重点关注最重要的元素，高效描述。 |
| "完整" | 所有相关项目 | 在合理范围内提供详尽覆盖，无重大遗漏，必要时使用分类。 |
| "彻底" | 7-10 | 提供详细覆盖，解决主要主题和子主题，兼具广度和深度。 |
| "概述" | 3-5 | 覆盖主要概念/方面，平衡覆盖，专注于基本理解。 |
| "总结" | 3-5个要点 | 提炼基本信息，简洁而全面地捕捉主要要点。 |
| "主要"或"关键" | 3-7 | 专注于对理解至关重要的最重要元素，涵盖不同方面。 |
| "基本" | 3-7 | 仅包括关键、必要项目，不包括外围或可选元素。 |
| "基础" | 2-5 | 呈现面向初学者的基础概念，专注于核心原则。 |
| "详细" | 5-10个带详细说明 | 提供深入覆盖，包括列表之外的解释，包括具体信息和细微差别。 |
| "常见" | 4-8个最频繁 | 专注于典型或普遍项目，尽可能按频率排序，这些项目被广泛认可。 |
| "主要" | 2-5个最重要 | 专注于主导因素，解释其首要性和超大影响。 |
| "次要" | 3-7个支持项目 | 呈现重要但非关键的项目，补充主要因素并提供额外上下文。 |
| 未指定分析 | 3-5个要点 | 默认为3-5个主要点，涵盖主要方面，平衡广度和深度。 |
</规则>
`,
    user:
      `
问题: ${question}
回答: ${answer}

请查看我的回答并思考。
`
  }
}


function getQuestionEvaluationPrompt(question: string): PromptPair {
  return {
    system: `You are an evaluator that determines if a question requires definitive, freshness, plurality, and/or completeness checks.

<evaluation_types>
definitive - Checks if the question requires a definitive answer or if uncertainty is acceptable (open-ended, speculative, discussion-based)
freshness - Checks if the question is time-sensitive or requires very recent information
plurality - Checks if the question asks for multiple items, examples, or a specific count or enumeration
completeness - Checks if the question explicitly mentions multiple named elements that all need to be addressed
</evaluation_types>

<rules>
1. Definitive Evaluation:
   - Required for ALMOST ALL questions - assume by default that definitive evaluation is needed
   - Not required ONLY for questions that are genuinely impossible to evaluate definitively
   - Examples of impossible questions: paradoxes, questions beyond all possible knowledge
   - Even subjective-seeming questions can be evaluated definitively based on evidence
   - Future scenarios can be evaluated definitively based on current trends and information
   - Look for cases where the question is inherently unanswerable by any possible means

2. Freshness Evaluation:
   - Required for questions about current state, recent events, or time-sensitive information
   - Required for: prices, versions, leadership positions, status updates
   - Look for terms: "current", "latest", "recent", "now", "today", "new"
   - Consider company positions, product versions, market data time-sensitive

3. Plurality Evaluation:
   - ONLY apply when completeness check is NOT triggered
   - Required when question asks for multiple examples, items, or specific counts
   - Check for: numbers ("5 examples"), list requests ("list the ways"), enumeration requests
   - Look for: "examples", "list", "enumerate", "ways to", "methods for", "several"
   - Focus on requests for QUANTITY of items or examples

4. Completeness Evaluation:
   - Takes precedence over plurality check - if completeness applies, set plurality to false
   - Required when question EXPLICITLY mentions multiple named elements that all need to be addressed
   - This includes:
     * Named aspects or dimensions: "economic, social, and environmental factors"
     * Named entities: "Apple, Microsoft, and Google", "Biden and Trump"
     * Named products: "iPhone 15 and Samsung Galaxy S24"
     * Named locations: "New York, Paris, and Tokyo"
     * Named time periods: "Renaissance and Industrial Revolution"
   - Look for explicitly named elements separated by commas, "and", "or", bullets
   - Example patterns: "comparing X and Y", "differences between A, B, and C", "both P and Q"
   - DO NOT trigger for elements that aren't specifically named   
</rules>

<examples>
<example-1>
谁发明了微积分？牛顿和莱布尼兹各自的贡献是什么？
<think>
这是关于微积分历史的问题，不涉及需要最新信息的内容。问题明确提到了牛顿和莱布尼兹两位数学家，要求分析他们各自的贡献，所以需要全面评估这两个特定的方面。这个问题涉及历史事实，有明确的学术研究可以参考，因此需要确定性评估。
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-1>

<example-2>
fam PLEASE help me calculate the eigenvalues of this 4x4 matrix ASAP!! [matrix details] got an exam tmrw 😭
<think>
This is a mathematical question about eigenvalues which doesn't change over time, so no need for recent information. A 4x4 matrix has multiple eigenvalues, so this requires identifying several distinct values. This is a pure mathematics problem with precise, verifiable solutions that can be definitively evaluated. The question asks for calculation of eigenvalues only, not addressing multiple distinct topics.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-2>

<example-3>
Quelles sont les principales différences entre le romantisme et le réalisme dans la littérature du 19ème siècle?
<think>
C'est une question sur l'histoire littéraire, donc aucun besoin d'informations récentes. La question mentionne spécifiquement deux mouvements: le romantisme et le réalisme. Je dois évaluer ces deux éléments nommés, donc l'exhaustivité est importante ici. Cette question porte sur des concepts littéraires établis avec des caractéristiques documentées, donc une évaluation définitive est possible. La question ne demande pas une liste ou énumération multiple au-delà des deux mouvements spécifiés.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-3>

<example-4>
Shakespeare の最も有名な悲劇を5つ挙げ、簡単にあらすじを説明してください。
<think>
シェイクスピアの悲劇についての質問であり、時事的な情報は不要。「5つ挙げ」という指定があるため、複数の項目が求められている。「最も有名な」という基準は学術的コンセンサスや文化的重要性に基づいて判断できるため、確定的な評価が可能。特定の作品を分析するよう求められているわけではなく、複数の作品を列挙することが主な要件。
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-4>

<example-5>
What are the current interest rates for mortgage loans from Bank of America, Wells Fargo, and Chase Bank in the US?
<think>
This question asks about "current" interest rates, so it clearly requires up-to-date information. The query specifically names three banks: Bank of America, Wells Fargo, and Chase Bank. Each of these named entities must be addressed, making completeness necessary. This question seeks factual financial data that can be objectively verified, so definitive evaluation is needed. The question isn't asking for multiple types of information beyond the specified banks.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-5>

<example-6>
2025年に注目すべき人工知能の3つのトレンドは何ですか？
<think>
これは将来のAIトレンドに関する質問なので、最新の情報が必要。「3つの」という明確な数を指定しているため、複数の項目が求められている。未来の予測だが、現在のAI開発動向や研究から推測できる範囲であり、確定的な評価が可能。特定の側面について言及しておらず、トレンドの列挙が主な要件なので、複数性のほうが重要。
</think>
<output>
"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-6>

<example-7>
Was sind die besten Strategien für nachhaltiges Investieren in der heutigen Wirtschaft?
<think>
Diese Frage bezieht sich auf die "heutige Wirtschaft", daher sind aktuelle Informationen erforderlich. "Strategien" steht im Plural, was auf die Notwendigkeit mehrerer Beispiele hindeutet. Obwohl "beste" subjektiv klingen mag, lässt sich die Frage anhand von Rendite-Daten, Risikobewertungen und Nachhaltigkeitskriterien definitiv evaluieren. Es werden keine spezifischen Aspekte genannt, die alle behandelt werden müssen - der Fokus liegt auf der Vielfalt der Strategien.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-7>

<example-8>
请解释赤壁之战的历史背景、主要参与者以及战略意义，这对中国历史产生了什么影响？
<think>
这是关于历史事件的问题，不需要最新信息。问题明确列出了多个需要评估的方面：历史背景、主要参与者、战略意义和历史影响。这些都是特定的历史课题，需要完整地覆盖。关于历史事件的问题有明确的史料记载和学术研究，可以进行确定性评估。
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": true,
</output>
</example-8>

<example-9>
How might artificial intelligence transform the healthcare industry over the next decade?
<think>
This question asks about future developments that require recent AI and healthcare information to properly evaluate. The query is about transformations (plural) suggesting multiple changes should be identified. While it concerns future trends, the patterns can be evaluated definitively based on current research, development pipelines, and industry analyses. No specific aspects of healthcare are mentioned that must all be addressed, so plurality is more relevant than completeness.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": true,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-9>

<example-10>
What do you think is the best approach to learning a new language as an adult?
<think>
This question about language learning does not require time-sensitive information as fundamental language acquisition principles remain relatively stable. The phrasing suggests looking for multiple elements or methods that constitute an effective approach, indicating plurality. Though the question uses subjective language ("what do you think"), it can be evaluated definitively using linguistic research and proven methodologies. No specific language learning aspects are explicitly named that must all be covered.
</think>
<output>
"needsDefinitive": true,
"needsFreshness": false,
"needsPlurality": true,
"needsCompleteness": false,
</output>
</example-10>

<example-11>
If a tree falls in a forest with absolutely no observers, instruments, or any possible way to detect it, does it make a sound?
<think>
This is a classic philosophical paradox that is inherently unanswerable in a definitive way. The question deliberately constructs a scenario that removes all possible means of verification, making it logically impossible to evaluate. This kind of question represents one of the rare cases where a definitive evaluation is truly impossible. The question doesn't involve recent events, doesn't request multiple items, and doesn't specify multiple elements that must be addressed.
</think>
<output>
"needsDefinitive": false,
"needsFreshness": false,
"needsPlurality": false,
"needsCompleteness": false,
</output>
</example-11>
</examples>

`,
    user:
      `
${question}
<think>`
  };
}

export async function evaluateQuestion(
  question: string,
  trackers: TrackerContext,
  schemaGen: Schemas
): Promise<EvaluationType[]> {
  try {
    const generator = new ObjectGeneratorSafe(trackers.tokenTracker);
    const prompt = getQuestionEvaluationPrompt(question);

    const result = await generator.generateObject({
      model: TOOL_NAME,
      schema: schemaGen.getQuestionEvaluateSchema(),
      system: prompt.system,
      prompt: prompt.user
    });



    // Always include definitive in types
    const types: EvaluationType[] = [];
    if (result.object.needsDefinitive) types.push('definitive');
    if (result.object.needsFreshness) types.push('freshness');
    if (result.object.needsPlurality) types.push('plurality');
    if (result.object.needsCompleteness) types.push('completeness');

    logInfo(TOOL_NAME, { question, types });
    trackers?.actionTracker.trackThink(result.object.think);

    // Always evaluate definitive first, then freshness (if needed), then plurality (if needed)
    return types;

  } catch (error) {
    logError('Error in question evaluation:', { error });
    // Default to no check
    return [];
  }
}


async function performEvaluation<T>(
  evaluationType: EvaluationType,
  prompt: PromptPair,
  trackers: TrackerContext,
  schemaGen: Schemas
): Promise<GenerateObjectResult<T>> {
  const generator = new ObjectGeneratorSafe(trackers.tokenTracker);
  const result = await generator.generateObject({
    model: TOOL_NAME,
    schema: schemaGen.getEvaluatorSchema(evaluationType),
    system: prompt.system,
    prompt: prompt.user
  }) as GenerateObjectResult<any>;

  trackers.actionTracker.trackThink(result.object.think)

  logInfo(`${evaluationType} ${TOOL_NAME}`, result.object);

  return result;
}


// Main evaluation function
export async function evaluateAnswer(
  question: string,
  action: AnswerAction,
  evaluationTypes: EvaluationType[],
  trackers: TrackerContext,
  allKnowledge: KnowledgeItem[],
  schemaGen: Schemas
): Promise<EvaluationResponse> {
  let result;


  for (const evaluationType of evaluationTypes) {
    let prompt: { system: string; user: string } | undefined
    switch (evaluationType) {

      case 'definitive':
        prompt = getDefinitivePrompt(question, action.answer);
        break;
      case 'freshness':
        prompt = getFreshnessPrompt(question, action, new Date().toISOString());
        break;
      case 'plurality':
        prompt = getPluralityPrompt(question, action.answer);
        break;
      case 'completeness':
        prompt = getCompletenessPrompt(question, action.answer);
        break;
      case 'strict':
        prompt = getRejectAllAnswersPrompt(question, action, allKnowledge);
        break;
      default:
        logError(`Unknown evaluation type: ${evaluationType}`);
    }
    if (prompt) {
      try {
        result = await performEvaluation(
          evaluationType,
          prompt,
          trackers,
          schemaGen
        );
      } catch (error) {
        logError(`Error performing ${evaluationType} evaluation`, { error });
        return { pass: false, think: `Error ${evaluationType} immedidately return false, probably due to bad prompt?`, type: evaluationType } as EvaluationResponse
      }

      // fail one, return immediately
      if (!(result?.object as EvaluationResponse)?.pass) {
        return result?.object as EvaluationResponse;
      }
    }
  }

  return result?.object as EvaluationResponse;

}
