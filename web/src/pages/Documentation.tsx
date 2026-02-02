import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import {
    Box,
    Typography,
    List,
    ListItemButton,
    ListItemText,
    ListItemIcon,
    TextField,
    InputAdornment,
    IconButton,
    Button
} from '@mui/material';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import StorageIcon from '@mui/icons-material/Storage';
import ExtensionIcon from '@mui/icons-material/Extension';
import PsychologyIcon from '@mui/icons-material/Psychology';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import GavelIcon from '@mui/icons-material/Gavel';
import SearchIcon from '@mui/icons-material/Search';
import CloseIcon from '@mui/icons-material/Close';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import HomeIcon from '@mui/icons-material/Home';
import AccountBalanceWalletIcon from '@mui/icons-material/AccountBalanceWallet';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// --- Content Data (Markdown) ---

const CONTENT_QUICK_START = `
## 配置 API 密钥

本系统使用 Google Gemini 或 OpenAI 作为 AI 分析引擎。您需要先获取相应的 API 密钥才能使用 AI 分析功能。

**步骤 1: 获取 Gemini API 密钥**

访问 [Google AI Studio](https://aistudio.google.com/app/apikey) 创建您的 API 密钥。Gemini API 提供免费额度，适合个人用户使用。

**步骤 2: 配置密钥到系统**

进入 **系统设置** 页面，在 LLM 配置区域选择提供商并填入您的 API 密钥。

\`\`\`
系统设置 → LLM 引擎 → 选择 Gemini → 填入 API Key → 保存
\`\`\`

**步骤 3: 验证配置**

保存后，系统会自动验证 API 密钥是否有效。您可以在 AI 推荐或报告页面尝试生成内容来确认配置成功。

> **提示：** Gemini API 每分钟有 15 次免费调用额度。如果需要更高频率，可以升级付费版或切换使用 OpenAI API。

---

## 创建投资组合

系统支持多投资组合管理，您可以创建多个组合来管理不同的投资策略：

1. **进入组合页面** - 点击左侧导航栏的"投资组合"
2. **创建新组合** - 点击"新建组合"按钮，输入组合名称
3. **添加持仓** - 在组合中添加股票或基金持仓，记录买入价格和数量
4. **设为默认** - 可以将常用组合设为默认组合

---

## 浏览市场数据

系统启动后会自动从 Akshare 获取市场数据。您可以通过以下方式浏览：

1. **仪表盘** - 查看市场指数、资金流向、板块表现、异动提醒等实时数据
2. **新闻资讯** - 浏览最新财经新闻，支持按分类筛选和关键词搜索
3. **基金/股票管理** - 搜索并添加您关注的标的到自选列表

---

## 使用 AI 功能

配置好 API 密钥后，您可以使用以下 AI 功能：

- **AI 智能推荐** - 进入"AI 精选"页面，基于量化因子模型 + LLM 分析，获取股票和基金推荐
- **投资分析报告** - 在"情报"页面为基金或股票生成盘前/盘后分析报告
- **组合诊断** - 在投资组合页面使用 AI 诊断功能，获取组合健康度分析和调仓建议
- **压力测试** - 使用 AI 生成市场情景，测试组合在不同市场环境下的表现
`;

const CONTENT_DATA_SOURCES = `
## 数据源概览

本系统整合多个数据源，提供全面的金融市场数据支持：

| 数据源 | 用途 | 说明 |
| :--- | :--- | :--- |
| **Akshare** | 核心市场数据 | A股行情、基金净值、资金流向、板块数据 |
| **TuShare** | 新闻资讯 | 财经新闻、公告、研报等信息流 |
| **YFinance** | 国际市场 | 黄金、白银等国际商品价格 |
| **Tavily** | 网络搜索 | 实时新闻搜索和情绪分析（可选） |

---

## Akshare 数据源

[Akshare](https://akshare.akfamily.xyz/) 是系统的核心数据源，提供 A 股、港股、美股等多市场的全面数据支持。

| 属性 | 说明 |
| :--- | :--- |
| **数据来源** | 东方财富、新浪财经、同花顺等 |
| **更新频率** | 实时行情 / 每日收盘更新 |
| **数据类型** | 行情、财务、资金流向、板块 |
| **覆盖范围** | A股、港股、基金、期货、外汇 |

---

## 市场行情数据

系统获取的市场行情数据包括以下核心指标：

- **股票实时行情**：最新价、涨跌幅、成交量、成交额、换手率、振幅
- **历史 K 线数据**：日线、周线历史走势，支持技术分析
- **资金流向数据**：主力净流入、散户净流入、大单成交占比、北向资金
- **估值指标**：市盈率(PE)、市净率(PB)、市销率(PS)、总市值、流通市值
- **财务数据**：ROE、净利润增长率、营收增长率、资产负债率等
- **基金净值**：单位净值、累计净值、日增长率、历史净值走势
- **融资融券**：融资余额、融券余额、融资买入额

---

## 行业板块数据

系统覆盖申万一级行业分类，支持 31 个行业板块的数据分析：

科技、医药生物、食品饮料、银行、非银金融、房地产、电力设备、汽车、机械设备、电子、通信、计算机、传媒、国防军工、建筑材料、建筑装饰、化工、有色金属、钢铁、煤炭、石油石化、公用事业、交通运输、农林牧渔、家用电器、轻工制造、纺织服饰、商贸零售、社会服务、美容护理、环保

---

## 基金数据说明

系统支持多种类型基金的数据获取和分析：

| 基金类型 | 数据涵盖 |
| :--- | :--- |
| **股票型基金** | 净值走势、重仓股、行业配置、基金经理 |
| **混合型基金** | 资产配置比例、历史业绩、风险指标 |
| **指数型基金** | 跟踪指数、跟踪误差、规模变化 |
| **ETF 基金** | 实时价格、折溢价率、成交量 |
| **债券型基金** | 久期、到期收益率、信用评级分布 |
| **QDII 基金** | 投资区域、汇率影响、海外持仓 |

---

## 新闻资讯数据

系统通过 TuShare 获取多类型财经资讯：

- **实时快讯**：7x24 小时滚动财经快讯
- **早盘必读**：每日开盘前市场要闻汇总
- **公司公告**：上市公司重大事项公告
- **研究报告**：券商研报摘要和评级变动
- **宏观数据**：经济指标发布和政策解读
`;

const CONTENT_MODULES = `
## 仪表盘

仪表盘是系统的核心入口，采用可视化组件布局，提供市场全景概览。

| 组件 | 功能说明 |
| :--- | :--- |
| **市场指数** | 上证、深证、创业板等主要指数实时行情 |
| **市场情绪** | 综合多维度指标的市场情绪判断 |
| **系统状态** | 基金/股票数量、报告数量等统计 |
| **主力资金** | 主力资金净流入排行榜 |
| **板块表现** | 行业板块涨跌幅排行 |
| **北向资金** | 沪深港通资金流向分析 |
| **异动提醒** | 涨停、跌停、大单异动等实时提醒 |
| **涨跌榜** | 涨幅榜、跌幅榜、换手率榜等 |

> **使用技巧：** 仪表盘数据每分钟自动刷新，点击刷新按钮可手动更新。

---

## 投资组合

投资组合模块提供机构级的组合管理和分析功能，支持多组合管理。

**组合管理**
- 多组合支持：创建多个独立组合，管理不同投资策略
- 持仓管理：添加/编辑/删除持仓，记录交易历史
- 实时估值：自动获取最新价格，计算组合市值和收益

**收益分析**
- 收益概览：总收益、年化收益、夏普比率等核心指标
- 收益日历：日/月/年维度的收益热力图
- 收益明细：每日收益详情，支持 AI 解读涨跌原因

**风险分析**
- 相关性分析：持仓相关性热力图，评估分散化程度
- 压力测试：预设情景和自定义情景测试
- AI 情景生成：使用 AI 生成市场情景进行压力测试

**AI 功能**
- 组合诊断：AI 分析组合健康度，识别风险点
- 调仓建议：基于市场环境的智能调仓推荐
- 持仓信号：每个持仓的 AI 买卖信号

---

## 基金管理

基金管理模块用于管理基金自选列表，支持全市场公募基金的搜索和分析。

**核心功能**
- 智能搜索：支持基金代码、名称、拼音首字母搜索
- 基金对比：多基金横向对比分析
- 净值走势：历史净值图表和回撤分析
- 持仓分析：重仓股、行业配置、持仓重叠度检测
- AI 诊断：基金健康度评估和投资建议

**操作说明**
1. 点击"添加基金"按钮打开搜索对话框
2. 输入基金代码或名称进行搜索
3. 点击搜索结果添加到列表
4. 在列表中可以查看详情、对比或删除

---

## 股票管理

股票管理模块用于管理自选股列表，支持 A 股全市场股票的搜索和分析。

**核心功能**
- 全市场搜索：覆盖沪深北三大交易所全部股票
- 实时行情：最新价、涨跌幅、成交量等实时数据
- K线图表：日线、周线等多周期 K 线走势
- 财务分析：利润表、资产负债表、现金流量表
- 股东分析：十大股东、机构持仓变动
- 融资融券：融资余额、融券余额趋势
- 量化分析：技术指标、动量信号
- AI 诊断：个股健康度评估和投资建议

> **提示：** 股票数据存在约 15-30 分钟延迟，这是免费数据源的限制。

---

## AI 精选

AI 精选模块是系统的核心功能，通过量化因子模型 + LLM 分析，从全市场中精选优质标的。

**股票推荐**
- 短期推荐：关注动量和资金流向，适合短线交易
- 长期推荐：关注基本面和估值，适合价值投资

**基金推荐**
- Alpha 策略：追求超额收益的主动管理基金
- 动量策略：趋势跟踪型基金选择

点击设置图标可配置投资偏好：风险偏好、市值范围、行业偏好、估值限制等。

---

## 新闻资讯

新闻资讯模块提供个性化的财经新闻聚合服务。

**核心功能**
- 多源聚合：整合 TuShare 等多个新闻源
- 分类筛选：实时快讯、早盘必读、公司公告、研究报告
- 自选关联：根据自选股/基金筛选相关新闻
- 收藏管理：收藏重要新闻，方便后续查阅
- AI 分析：新闻情绪分析和影响评估
- 时间筛选：1天、3天、7天、全部时间范围

---

## 市场情绪

市场情绪模块通过多维度指标分析，综合判断当前市场情绪。

**分析维度**
- 乐观：市场情绪积极，可适当进取
- 中性：市场情绪平稳，宜观望为主
- 悲观：市场情绪低迷，注意风险控制

**参考指标**
- 涨跌家数比：上涨与下跌股票数量比值
- 涨停跌停数：涨停板和跌停板股票数量
- 成交量变化：与近期平均成交量对比
- 北向资金：外资流入流出情况
- 市场周期：当前所处市场周期阶段

---

## 情报中心

情报中心用于生成和管理 AI 分析报告。

**盘前分析报告**
- 隔夜全球市场回顾
- 今日重点关注事件
- 持仓标的开盘预判
- 今日操作建议

**盘后复盘报告**
- 当日市场走势回顾
- 持仓标的表现分析
- 板块轮动情况
- 下一交易日展望

**报告管理**
- 按日期分组展示，支持搜索
- 支持导出为图片
- 可删除不需要的历史报告

---

## 商品分析

商品分析模块提供贵金属等大宗商品的行情和分析。

**支持商品**
- 黄金：国际金价走势和技术分析
- 白银：国际银价走势和技术分析

**主要功能**
- 实时价格走势图表
- 技术指标分析（RSI、均线等）
- AI 生成商品分析报告
- 历史报告查看和管理

---

## 系统设置

系统设置模块用于配置系统核心参数。

**LLM 引擎配置**

| 配置项 | 说明 |
| :--- | :--- |
| LLM 提供商 | Gemini / OpenAI / OpenAI 兼容（自定义） |
| API Key | 对应提供商的 API 密钥 |
| Base URL | 自定义 API 端点（OpenAI 兼容模式） |
| 模型选择 | 选择或输入模型名称 |

**数据源配置**

| 配置项 | 说明 |
| :--- | :--- |
| Tavily API Key | 用于网络搜索和新闻情绪分析（可选） |

**邮件通知配置**

| 配置项 | 说明 |
| :--- | :--- |
| SMTP 服务器 | 邮件服务器地址和端口 |
| 发件人设置 | 发件邮箱和认证信息 |
| 收件人 | 接收通知的邮箱地址 |
| 通知触发 | 报告生成、异常提醒、每日摘要 |
| 免打扰时段 | 设置不发送通知的时间段 |

> **安全提示：** API 密钥是敏感信息，请妥善保管。系统会对密钥进行脱敏显示。
`;

const CONTENT_AI_LOGIC = `
## 推荐系统架构

AI 精选采用"量化因子筛选 + LLM 深度分析"的双层架构：

\`\`\`
第一层：量化因子模型
├── 数据采集 → 全市场股票/基金数据
├── 因子计算 → 动量、估值、质量、资金等因子
├── 候选筛选 → 根据因子得分筛选候选池
└── 输出候选池（约 50-100 只）

第二层：LLM 深度分析
├── 输入候选池 + 市场环境数据
├── AI 综合分析 → 基本面、技术面、市场情绪
├── 生成推荐理由和风险提示
└── 输出最终推荐（约 5-10 只）
\`\`\`

---

## 股票评分体系

股票推荐采用多维度评分体系，满分 100 分：

| 维度 | 权重 | 说明 |
| :--- | :--- | :--- |
| **动量得分** | 25% | 价格趋势强度、均线排列、突破信号 |
| **资金得分** | 25% | 主力净流入、大单占比、资金持续性 |
| **估值得分** | 20% | PE/PB 历史百分位、同行业对比 |
| **基本面得分** | 20% | ROE、盈利增长、营收增长、行业地位 |
| **情绪得分** | 10% | 市场热度、板块轮动、新闻舆情 |

---

## 基金评分体系

基金推荐关注以下核心因子：

| 维度 | 说明 |
| :--- | :--- |
| **业绩表现** | 近 1 周、1 月、3 月、1 年收益率 |
| **风险调整收益** | 夏普比率、最大回撤、波动率 |
| **规模因子** | 基金规模适中（避免过大或过小） |
| **经理能力** | 基金经理历史业绩和从业年限 |
| **持仓质量** | 重仓股质量和行业配置合理性 |

---

## 短期策略逻辑

> **投资周期：** 7-30 天 | **核心关注：** 动量和资金流向

短期策略适合捕捉市场短线交易机会，优先考虑以下因素：

- 资金持续流入：连续 3 日以上主力净流入
- 所属板块热度上升：板块整体表现强势
- 技术形态良好：放量突破、均线多头排列、MACD 金叉
- 有近期催化剂：业绩预告、政策利好、新产品发布
- 量比 > 1.5：交易活跃度提升

\`\`\`
筛选条件示例：
主力净流入 > 0 AND 量比 > 1.5 AND 板块热度排名 < 20
\`\`\`

---

## 长期策略逻辑

> **投资周期：** 3 个月以上 | **核心关注：** 基本面和估值

长期策略适合价值投资，优先考虑以下因素：

- 行业龙头或细分领域冠军：市场份额领先，竞争壁垒高
- 估值处于历史合理区间：PE/PB 百分位 < 60%
- ROE > 15%：盈利能力强，或有明显改善趋势
- 清晰的业绩增长逻辑：行业景气、产品升级、产能扩张
- 护城河明显：品牌优势、技术专利、规模效应

\`\`\`
筛选条件示例：
ROE > 15% AND PE百分位 < 60% AND 市值 > 100亿
\`\`\`

---

## 个性化偏好设置

系统支持根据您的投资偏好进行个性化推荐：

**投资画像**

| 偏好类型 | 选项 |
| :--- | :--- |
| 风险偏好 | 保守型 / 稳健型 / 积极型 / 投机型 |
| 投资周期 | 短期 / 中期 / 长期 |
| 投资目标 | 稳健增值 / 资本增长 / 高风险高收益 |

**筛选条件**

| 偏好类型 | 说明 |
| :--- | :--- |
| 市值范围 | 设置最小/最大市值范围 |
| 估值限制 | 设置 PE/PB 范围限制 |
| 行业偏好 | 设置偏好行业和排除行业 |
| 基金类型 | 选择偏好的基金类型 |
| 特殊规则 | 排除 ST 股票、排除次新股、要求盈利等 |

**风控参数**

| 参数 | 说明 |
| :--- | :--- |
| 最大回撤容忍 | 可接受的最大亏损比例 |
| 止损比例 | 单笔投资止损线 |
| 止盈比例 | 单笔投资止盈线 |
| 单笔仓位上限 | 单只标的最大仓位占比 |

---

## 组合诊断逻辑

AI 组合诊断从以下维度分析组合健康度：

| 维度 | 分析内容 |
| :--- | :--- |
| **收益分析** | 总收益、年化收益、相对基准超额收益 |
| **风险分析** | 波动率、最大回撤、VaR、夏普比率 |
| **集中度分析** | 单一持仓占比、行业集中度、相关性 |
| **估值分析** | 组合整体估值水平、高估/低估持仓 |
| **动量分析** | 持仓趋势强度、技术形态 |

诊断结果包括：
- 组合健康度评分（0-100）
- 主要风险点识别
- 调仓建议和优化方向
`;

const CONTENT_FAQ = `
## API 相关问题

**429 错误处理**

> **错误代码 429** 表示 API 请求频率超过限制（Rate Limit Exceeded）

常见原因：
- Gemini API 免费版每分钟 15 次请求限制
- 短时间内连续生成多份报告
- Akshare 数据接口调用过于频繁

解决方案：
- 等待 1-2 分钟后重试
- 升级到 Gemini API 付费版本
- 切换使用 OpenAI API（按用量计费）
- 减少并发请求

**API 调用限制**

| API 提供商 | 免费额度 | 限制说明 |
| :--- | :--- | :--- |
| Gemini | 15 RPM | 每分钟 15 次请求，每日有总量限制 |
| OpenAI | 按用量计费 | 无免费额度，按 token 数量计费 |
| Akshare | 无明确限制 | 建议请求间隔 1 秒以上 |

---

## 数据相关问题

**Q: 为什么数据不是实时更新的？**

A: Akshare 获取的行情数据存在约 15-30 分钟延迟，这是免费数据源的通用限制。如需实时行情，需要接入付费行情数据源。

**Q: 基金净值为什么是昨天的？**

A: 基金净值在每个交易日收盘后约 19:00-22:00 才会更新。如果当前时间早于净值更新时间，显示的是上一个交易日的净值数据。

**Q: 为什么搜索不到某些股票/基金？**

A: 请确保输入正确的代码或名称。新上市的股票/基金可能数据源尚未收录，通常在上市后 1-2 天内会更新。

---

## 功能使用问题

**Q: AI 分析报告生成失败怎么办？**

A: 请检查：
1. API 密钥是否正确配置
2. 网络连接是否正常
3. 是否触发了调用频率限制

如持续失败，尝试切换 LLM 提供商。

**Q: 如何创建多个投资组合？**

A: 进入"投资组合"页面，点击组合名称旁的下拉菜单，选择"新建组合"。您可以创建多个组合来管理不同的投资策略。

**Q: 如何使用压力测试功能？**

A: 在投资组合页面，点击"压力测试"标签页。您可以：
1. 选择预设情景（如市场崩盘、板块轮动等）
2. 使用滑块自定义市场参数
3. 点击"AI 生成情景"让 AI 创建测试场景

**Q: 如何配置邮件通知？**

A: 进入"系统设置"页面，切换到"邮件通知"标签页：
1. 开启邮件通知开关
2. 配置 SMTP 服务器信息
3. 设置收件人邮箱
4. 选择需要通知的事件类型
5. 点击"发送测试邮件"验证配置

**Q: 新闻资讯如何筛选自选股相关新闻？**

A: 在新闻页面，点击"自选关联"筛选器，系统会自动筛选出与您自选股/基金相关的新闻。

---

## 组合管理问题

**Q: 如何记录交易历史？**

A: 在投资组合中添加或修改持仓时，系统会自动记录交易历史。您可以在持仓详情中查看该持仓的所有交易记录。

**Q: 收益计算是如何进行的？**

A: 系统根据您记录的买入成本和当前市价计算收益。支持以下收益指标：
- 持仓收益 = (当前市值 - 成本) / 成本
- 总收益 = 所有持仓收益加权平均
- 年化收益 = 根据持仓时间折算的年化收益率

**Q: 相关性分析有什么用？**

A: 相关性分析帮助您了解持仓之间的关联程度。高相关性意味着持仓可能同涨同跌，分散化效果较差。建议保持持仓之间的低相关性以降低组合风险。

---

## 其他问题

**Q: 如何清除历史数据重新开始？**

A: 您可以在基金/股票管理页面逐个删除已添加的标的。投资组合可以在组合管理中删除。

**Q: 系统支持哪些语言？**

A: 系统支持中文和英文两种语言。点击左侧导航栏底部的语言切换按钮即可切换。

**Q: 如何导出报告？**

A: 在情报中心查看报告时，点击右上角的"导出"按钮，可以将报告导出为图片格式。
`;

const CONTENT_PORTFOLIO = `
## 组合管理概述

投资组合模块提供机构级的组合管理功能，帮助您系统化地管理投资。

**多组合支持**
- 创建多个独立投资组合
- 每个组合独立计算收益和风险
- 可设置默认组合
- 组合间数据完全隔离

**持仓管理**
- 添加股票或基金持仓
- 记录买入价格、数量、日期
- 支持多次加仓记录
- 自动计算持仓成本和收益

---

## 收益分析模块

收益分析提供全面的组合收益追踪和分析功能。

**收益概览**

| 指标 | 说明 |
| :--- | :--- |
| 总收益 | 组合成立以来的累计收益 |
| 年化收益 | 折算为年化的收益率 |
| 夏普比率 | 风险调整后的收益指标 |
| 最大回撤 | 历史最大亏损幅度 |
| 波动率 | 收益的标准差 |

**收益日历**

以热力图形式展示每日收益：
- 日视图：查看每天的收益情况
- 月视图：按月汇总收益
- 年视图：按年汇总收益

颜色编码：绿色表示盈利，红色表示亏损，颜色深浅表示幅度大小。

**收益明细**
- 查看每日收益详情
- AI 解读当日涨跌原因
- 分析最佳/最差交易日

---

## 风险分析模块

**相关性分析**

相关性热力图展示持仓之间的关联程度：

| 相关系数 | 含义 |
| :--- | :--- |
| > 0.7 | 高度正相关，分散化效果差 |
| 0.3 ~ 0.7 | 中度相关 |
| -0.3 ~ 0.3 | 低相关，分散化效果好 |
| < -0.3 | 负相关，对冲效果 |

系统会计算分散化评分，并提供 AI 解读。

**压力测试**

压力测试帮助您了解组合在极端市场环境下的表现。

预设情景：
- 市场崩盘（-20%）
- 板块轮动
- 利率上升
- 通胀冲击
- 流动性危机

自定义情景：使用滑块调整各项市场参数，创建自定义测试情景。

AI 情景生成：描述您想测试的市场环境，AI 会自动生成对应的参数设置。

---

## AI 诊断功能

**组合诊断**

AI 会从以下维度分析您的组合：

1. 收益表现 - 与基准对比，识别超额收益来源
2. 风险暴露 - 分析行业、风格、因子暴露
3. 集中度风险 - 检查单一持仓和行业集中度
4. 估值水平 - 评估组合整体估值是否合理
5. 动量状态 - 分析持仓的趋势强度

诊断结果包括健康度评分（0-100）和具体改进建议。

**调仓建议**

基于当前市场环境和组合状态，AI 会提供：
- 建议增持的标的
- 建议减持的标的
- 建议新增的标的
- 调仓理由和风险提示

**持仓信号**

每个持仓会显示 AI 信号：
- 买入信号 - 建议加仓
- 持有信号 - 建议维持
- 卖出信号 - 建议减仓

点击信号可查看详细分析理由。

---

## 使用建议

**组合构建**
1. 根据投资目标确定资产配置比例
2. 选择低相关性的标的以分散风险
3. 控制单一持仓占比（建议不超过 20%）
4. 定期检查组合健康度

**风险控制**
1. 设置止损线，严格执行
2. 关注最大回撤，及时调整
3. 利用压力测试评估极端风险
4. 保持适度分散化

**定期复盘**
1. 每周查看收益分析
2. 每月进行 AI 诊断
3. 根据市场变化调整配置
4. 记录投资决策和理由
`;

const CONTENT_DISCLAIMER = `
## 投资风险提示

- 股票、基金等金融产品存在市场风险，价格可能大幅波动，甚至可能损失全部本金
- 过往业绩不代表未来表现，历史收益率不能保证未来收益
- 投资者应根据自身风险承受能力、投资经验和财务状况做出独立决策
- 请勿将全部资产投入单一产品，建议分散投资以降低风险
- 建议在做出重大投资决策前咨询专业的持牌财务顾问

---

## AI 分析局限性

AI 分析系统存在以下固有局限性，请务必了解：

- AI 模型基于历史数据训练，无法预测突发事件（黑天鹅）的影响
- 市场受政策变化、地缘政治、投资者情绪等非量化因素影响
- 大语言模型可能产生"幻觉"，生成看似合理但实际不准确的分析
- 推荐评分仅为参考指标，不能保证推荐标的一定会上涨
- 系统无法替代专业投资顾问的综合判断和个性化建议
- AI 对市场的理解基于训练数据，可能存在认知偏差

---

## 数据准确性

关于数据准确性的重要说明：

- 数据来源于第三方免费接口（Akshare），可能存在延迟或误差
- 系统尽力保证数据准确，但不对数据的完整性和及时性承担责任
- 重要投资决策请以官方数据为准（如交易所官网、基金公司官网）
- 财务数据来源于公开披露的定期报告，可能未反映最新情况
- 如发现数据明显异常，请及时反馈，我们将尽快核实处理

---

## 法律声明

使用本系统即表示您已阅读、理解并同意以下条款：

1. 本系统提供的所有信息、分析和推荐仅供参考和学习交流目的，不构成投资建议、财务建议或任何形式的投资推荐。

2. 用户基于本系统信息做出的任何投资决策及其后果，风险由用户自行承担，系统开发者和运营者不承担任何责任。

3. 本系统不对因使用系统信息而导致的任何直接或间接损失（包括但不限于投资亏损、数据丢失、业务中断等）承担赔偿责任。

4. 用户应遵守所在国家和地区的法律法规，合法、合规地使用本系统，不得将系统用于任何非法目的。

5. 本系统保留随时修改、更新或终止服务的权利，以及修改本免责声明条款的权利。
`;

// --- Configuration ---

const DOC_SECTIONS = [
    { id: 'quick-start', title: '快速开始', titleEn: 'Quick Start', icon: RocketLaunchIcon, content: CONTENT_QUICK_START },
    { id: 'data-sources', title: '数据说明', titleEn: 'Data Sources', icon: StorageIcon, content: CONTENT_DATA_SOURCES },
    { id: 'modules', title: '功能模块', titleEn: 'Modules', icon: ExtensionIcon, content: CONTENT_MODULES },
    { id: 'portfolio', title: '投资组合', titleEn: 'Portfolio', icon: AccountBalanceWalletIcon, content: CONTENT_PORTFOLIO },
    { id: 'ai-logic', title: 'AI 逻辑', titleEn: 'AI Logic', icon: PsychologyIcon, content: CONTENT_AI_LOGIC },
    { id: 'faq', title: '常见问题', titleEn: 'FAQ', icon: HelpOutlineIcon, content: CONTENT_FAQ },
    { id: 'disclaimer', title: '免责声明', titleEn: 'Disclaimer', icon: GavelIcon, content: CONTENT_DISCLAIMER },
];

// Generate slug from text
const generateSlug = (text: string) => {
    return text
        .toLowerCase()
        .replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
        .replace(/^-|-$/g, '');
};

export default function DocumentationPage() {
    const [activeSection, setActiveSection] = useState('quick-start');
    const [searchQuery, setSearchQuery] = useState('');
    const [toc, setToc] = useState<{ id: string; text: string; level: number }[]>([]);
    const [visibleHeaders, setVisibleHeaders] = useState<Set<string>>(new Set());
    const contentRef = useRef<HTMLDivElement>(null);
    const isScrollingRef = useRef(false);

    // Filtered sections based on search
    const filteredSections = useMemo(() => {
        if (!searchQuery) return DOC_SECTIONS;
        const q = searchQuery.toLowerCase();
        return DOC_SECTIONS.filter(s =>
            s.title.toLowerCase().includes(q) ||
            s.titleEn.toLowerCase().includes(q) ||
            s.content.toLowerCase().includes(q)
        );
    }, [searchQuery]);

    const activeDoc = useMemo(() =>
        DOC_SECTIONS.find(s => s.id === activeSection) || DOC_SECTIONS[0],
        [activeSection]);

    // Parse Markdown headers for Table of Contents
    useEffect(() => {
        const lines = activeDoc.content.split('\n');
        const headers = lines
            .filter(line => line.startsWith('#'))
            .map(line => {
                const level = line.match(/^#+/)?.[0].length || 0;
                const text = line.replace(/^#+\s+/, '').trim();
                const id = generateSlug(text);
                return { id, text, level };
            });
        setToc(headers);
        setVisibleHeaders(new Set());
    }, [activeDoc]);

    // Scroll Spy - track visible headers
    useEffect(() => {
        const handleScroll = () => {
            if (!contentRef.current || isScrollingRef.current) return;

            const container = contentRef.current;
            const containerRect = container.getBoundingClientRect();
            const headers = container.querySelectorAll('h2, h3');
            const newVisibleHeaders = new Set<string>();

            headers.forEach((header) => {
                const rect = header.getBoundingClientRect();
                // Check if header is in the visible area of the container
                const isVisible = rect.top >= containerRect.top - 100 && rect.top <= containerRect.bottom - 100;
                if (isVisible && header.id) {
                    newVisibleHeaders.add(header.id);
                }
            });

            // If no headers visible, find the one above the viewport
            if (newVisibleHeaders.size === 0) {
                let lastAbove = '';
                headers.forEach((header) => {
                    const rect = header.getBoundingClientRect();
                    if (rect.top < containerRect.top + 100 && header.id) {
                        lastAbove = header.id;
                    }
                });
                if (lastAbove) {
                    newVisibleHeaders.add(lastAbove);
                }
            }

            setVisibleHeaders(newVisibleHeaders);
        };

        const el = contentRef.current;
        if (el) {
            el.addEventListener('scroll', handleScroll, { passive: true });
            // Initial check
            setTimeout(handleScroll, 100);
            return () => el.removeEventListener('scroll', handleScroll);
        }
    }, [activeDoc]);

    const scrollToHeader = useCallback((id: string) => {
        const element = document.getElementById(id);
        if (element && contentRef.current) {
            isScrollingRef.current = true;

            const containerTop = contentRef.current.getBoundingClientRect().top;
            const elementTop = element.getBoundingClientRect().top;
            const offset = elementTop - containerTop + contentRef.current.scrollTop - 32;

            contentRef.current.scrollTo({ top: offset, behavior: 'smooth' });

            setTimeout(() => {
                isScrollingRef.current = false;
                setVisibleHeaders(new Set([id]));
            }, 500);
        }
    }, []);

    const handleSectionChange = useCallback((sectionId: string) => {
        setActiveSection(sectionId);
        setSearchQuery('');
        if (contentRef.current) {
            contentRef.current.scrollTo({ top: 0, behavior: 'instant' });
        }
    }, []);

    // Custom Components for ReactMarkdown with improved typography
    const components = useMemo(() => ({
        h1: ({ children, ...props }: any) => (
            <h1
                id={generateSlug(String(children))}
                className="text-3xl font-bold text-slate-900 mt-12 mb-6 tracking-tight leading-tight"
                {...props}
            >
                {children}
            </h1>
        ),
        h2: ({ children, ...props }: any) => (
            <h2
                id={generateSlug(String(children))}
                className="text-2xl font-bold text-slate-800 mt-16 mb-6 pb-3 border-b border-slate-200 leading-tight first:mt-0"
                {...props}
            >
                {children}
            </h2>
        ),
        h3: ({ children, ...props }: any) => (
            <h3
                id={generateSlug(String(children))}
                className="text-lg font-semibold text-slate-700 mt-10 mb-4 leading-snug"
                {...props}
            >
                {children}
            </h3>
        ),
        p: ({ children, ...props }: any) => (
            <p className="text-base text-slate-600 mb-6 leading-relaxed" {...props}>
                {children}
            </p>
        ),
        ul: ({ children, ...props }: any) => (
            <ul className="my-6 ml-6 space-y-3" {...props}>
                {children}
            </ul>
        ),
        ol: ({ children, ...props }: any) => (
            <ol className="my-6 ml-6 space-y-3 list-decimal" {...props}>
                {children}
            </ol>
        ),
        li: ({ children, ...props }: any) => (
            <li className="text-slate-600 leading-relaxed pl-2" {...props}>
                <span className="relative">
                    {children}
                </span>
            </li>
        ),
        blockquote: ({ children, ...props }: any) => (
            <blockquote
                className="my-8 pl-6 py-4 pr-6 border-l-4 border-indigo-400 bg-indigo-50/50 rounded-r-lg"
                {...props}
            >
                <div className="text-slate-700 text-sm leading-relaxed [&>p]:mb-0">
                    {children}
                </div>
            </blockquote>
        ),
        code: ({ inline, children, ...props }: any) => {
            if (inline) {
                return (
                    <code
                        className="px-1.5 py-0.5 bg-slate-100 text-indigo-600 text-sm font-mono rounded"
                        {...props}
                    >
                        {children}
                    </code>
                );
            }
            return (
                <div className="my-8 rounded-xl overflow-hidden bg-slate-900 shadow-lg">
                    <div className="flex items-center gap-2 px-4 py-3 bg-slate-800 border-b border-slate-700">
                        <div className="w-3 h-3 rounded-full bg-red-500" />
                        <div className="w-3 h-3 rounded-full bg-yellow-500" />
                        <div className="w-3 h-3 rounded-full bg-green-500" />
                        <span className="ml-3 text-xs text-slate-400 font-mono">terminal</span>
                    </div>
                    <pre className="p-5 overflow-x-auto">
                        <code className="text-emerald-400 font-mono text-sm leading-relaxed" {...props}>
                            {children}
                        </code>
                    </pre>
                </div>
            );
        },
        table: ({ children, ...props }: any) => (
            <div className="my-8 overflow-x-auto rounded-xl border border-slate-200 shadow-sm">
                <table className="w-full text-sm" {...props}>
                    {children}
                </table>
            </div>
        ),
        thead: ({ children, ...props }: any) => (
            <thead className="bg-slate-50 border-b border-slate-200" {...props}>
                {children}
            </thead>
        ),
        tbody: ({ children, ...props }: any) => (
            <tbody className="divide-y divide-slate-100" {...props}>
                {children}
            </tbody>
        ),
        tr: ({ children, ...props }: any) => (
            <tr className="hover:bg-slate-50/50 transition-colors" {...props}>
                {children}
            </tr>
        ),
        th: ({ children, ...props }: any) => (
            <th className="px-5 py-3.5 text-left font-semibold text-slate-700 whitespace-nowrap" {...props}>
                {children}
            </th>
        ),
        td: ({ children, ...props }: any) => (
            <td className="px-5 py-4 text-slate-600" {...props}>
                {children}
            </td>
        ),
        a: ({ children, ...props }: any) => (
            <a
                className="text-indigo-600 font-medium hover:text-indigo-700 hover:underline underline-offset-2 transition-colors"
                target="_blank"
                rel="noopener noreferrer"
                {...props}
            >
                {children}
            </a>
        ),
        hr: () => <hr className="my-12 border-t border-slate-200" />,
        strong: ({ children, ...props }: any) => (
            <strong className="font-semibold text-slate-800" {...props}>
                {children}
            </strong>
        ),
    }), []);

    const currentIdx = DOC_SECTIONS.findIndex(s => s.id === activeSection);
    const prevSection = DOC_SECTIONS[currentIdx - 1];
    const nextSection = DOC_SECTIONS[currentIdx + 1];

    return (
        <div className="flex h-screen bg-white">
            {/* Left Sidebar */}
            <aside className="w-72 flex-shrink-0 border-r border-slate-200 bg-slate-50 flex flex-col">
                {/* Header */}
                <div className="h-16 flex items-center justify-between px-5 border-b border-slate-200 bg-white">
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center">
                            <MenuBookIcon className="text-white" sx={{ fontSize: 18 }} />
                        </div>
                        <div>
                            <div className="text-sm font-bold text-slate-800 leading-none">VAlpha</div>
                            <div className="text-xs text-slate-400 mt-0.5">Documentation</div>
                        </div>
                    </div>
                    <IconButton
                        size="small"
                        onClick={() => window.location.href = '/'}
                        sx={{ color: '#64748b' }}
                    >
                        <HomeIcon fontSize="small" />
                    </IconButton>
                </div>

                {/* Search */}
                <div className="p-4 border-b border-slate-200 bg-white">
                    <TextField
                        fullWidth
                        placeholder="搜索文档..."
                        size="small"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        InputProps={{
                            startAdornment: (
                                <InputAdornment position="start">
                                    <SearchIcon sx={{ fontSize: 18, color: '#94a3b8' }} />
                                </InputAdornment>
                            ),
                            endAdornment: searchQuery && (
                                <InputAdornment position="end">
                                    <IconButton size="small" onClick={() => setSearchQuery('')}>
                                        <CloseIcon sx={{ fontSize: 16 }} />
                                    </IconButton>
                                </InputAdornment>
                            ),
                            sx: {
                                borderRadius: '10px',
                                fontSize: '0.875rem',
                                bgcolor: '#f8fafc',
                                '& fieldset': { borderColor: '#e2e8f0' },
                                '&:hover fieldset': { borderColor: '#cbd5e1' },
                            }
                        }}
                    />
                </div>

                {/* Navigation */}
                <nav className="flex-1 overflow-y-auto py-3 px-3">
                    <List disablePadding>
                        {filteredSections.map((section) => {
                            const isActive = activeSection === section.id;
                            const IconComponent = section.icon;
                            return (
                                <ListItemButton
                                    key={section.id}
                                    selected={isActive}
                                    onClick={() => handleSectionChange(section.id)}
                                    sx={{
                                        borderRadius: '10px',
                                        mb: 0.5,
                                        py: 1.25,
                                        px: 1.5,
                                        '&.Mui-selected': {
                                            bgcolor: '#eef2ff',
                                            '&:hover': { bgcolor: '#e0e7ff' },
                                        },
                                        '&:hover': { bgcolor: '#f1f5f9' }
                                    }}
                                >
                                    <ListItemIcon sx={{ minWidth: 36 }}>
                                        <IconComponent
                                            sx={{
                                                fontSize: 20,
                                                color: isActive ? '#4f46e5' : '#94a3b8'
                                            }}
                                        />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={section.title}
                                        secondary={section.titleEn}
                                        primaryTypographyProps={{
                                            fontSize: '0.875rem',
                                            fontWeight: isActive ? 600 : 500,
                                            color: isActive ? '#4f46e5' : '#475569'
                                        }}
                                        secondaryTypographyProps={{
                                            fontSize: '0.7rem',
                                            color: '#94a3b8',
                                            mt: 0.25
                                        }}
                                    />
                                </ListItemButton>
                            );
                        })}
                    </List>
                </nav>

                {/* Footer */}
                <div className="p-4 border-t border-slate-200 bg-white">
                    <Typography variant="caption" className="text-slate-400 block text-center">
                        v2.2 · VAlpha Terminal
                    </Typography>
                </div>
            </aside>

            {/* Main Content */}
            <main ref={contentRef} className="flex-1 overflow-y-auto">
                <div className="max-w-3xl mx-auto px-8 py-12">
                    {/* Breadcrumb */}
                    <div className="flex items-center gap-2 text-sm text-slate-400 mb-8">
                        <span>Docs</span>
                        <ChevronRightIcon sx={{ fontSize: 16 }} />
                        <span className="text-indigo-600 font-medium">{activeDoc.title}</span>
                    </div>

                    {/* Title */}
                    <div className="mb-12">
                        <h1 className="text-4xl font-bold text-slate-900 tracking-tight mb-3">
                            {activeDoc.title}
                        </h1>
                        <p className="text-lg text-slate-500">
                            {activeDoc.titleEn}
                        </p>
                    </div>

                    {/* Content */}
                    <article className="prose-doc">
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={components}
                        >
                            {activeDoc.content}
                        </ReactMarkdown>
                    </article>

                    {/* Navigation Footer */}
                    <div className="mt-20 pt-8 border-t border-slate-200 flex justify-between items-center">
                        {prevSection ? (
                            <Button
                                onClick={() => handleSectionChange(prevSection.id)}
                                startIcon={<ChevronLeftIcon />}
                                sx={{
                                    textTransform: 'none',
                                    color: '#64748b',
                                    '&:hover': { bgcolor: '#f1f5f9' }
                                }}
                            >
                                <Box className="text-left">
                                    <Typography variant="caption" className="block text-slate-400">
                                        上一章
                                    </Typography>
                                    <Typography className="font-medium text-slate-700">
                                        {prevSection.title}
                                    </Typography>
                                </Box>
                            </Button>
                        ) : <div />}
                        {nextSection && (
                            <Button
                                onClick={() => handleSectionChange(nextSection.id)}
                                endIcon={<ChevronRightIcon />}
                                sx={{
                                    textTransform: 'none',
                                    color: '#64748b',
                                    '&:hover': { bgcolor: '#f1f5f9' }
                                }}
                            >
                                <Box className="text-right">
                                    <Typography variant="caption" className="block text-slate-400">
                                        下一章
                                    </Typography>
                                    <Typography className="font-medium text-slate-700">
                                        {nextSection.title}
                                    </Typography>
                                </Box>
                            </Button>
                        )}
                    </div>
                </div>
            </main>

            {/* Right Sidebar - Table of Contents */}
            <aside className="w-56 flex-shrink-0 border-l border-slate-200 bg-white hidden xl:block">
                <div className="sticky top-0 p-5">
                    <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                        本页目录
                    </div>
                    <nav className="space-y-1">
                        {toc.map((item) => {
                            const isVisible = visibleHeaders.has(item.id);
                            const indent = (item.level - 2) * 12;
                            return (
                                <button
                                    key={item.id}
                                    onClick={() => scrollToHeader(item.id)}
                                    style={{ paddingLeft: indent + 12 }}
                                    className={`
                                        w-full text-left py-1.5 pr-3 text-sm rounded-r
                                        transition-all duration-150 relative
                                        ${isVisible
                                            ? 'text-slate-900 font-medium'
                                            : 'text-slate-400 hover:text-slate-600'
                                        }
                                    `}
                                >
                                    {/* Black indicator bar */}
                                    <span
                                        className={`
                                            absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-full
                                            transition-all duration-150
                                            ${isVisible ? 'bg-slate-800' : 'bg-transparent'}
                                        `}
                                    />
                                    <span className="line-clamp-1">{item.text}</span>
                                </button>
                            );
                        })}
                    </nav>
                </div>
            </aside>
        </div>
    );
}
