import axios from 'axios';

// In production (build), use relative path '/api' to let Nginx proxy handle it.
// In development (dev), use env var or fallback to localhost:8000.
const API_BASE = import.meta.env.PROD ? '/api' : (import.meta.env.VITE_API_URL || 'http://localhost:8000/api');

export interface ReportSummary {
  filename: string;
  date: string;
  mode: 'pre' | 'post' | 'commodities';
  fund_code?: string;
  fund_name?: string;
  is_summary: boolean;
}

export const fetchReports = async (): Promise<ReportSummary[]> => {
  const response = await axios.get(`${API_BASE}/reports`);
  return response.data;
};

export const deleteReport = async (filename: string): Promise<void> => {
  await axios.delete(`${API_BASE}/reports/${filename}`);
};

export const fetchCommodityReports = async (): Promise<ReportSummary[]> => {
  const response = await axios.get(`${API_BASE}/commodities/reports`);
  return response.data;
};

export const deleteCommodityReport = async (filename: string): Promise<void> => {
  await axios.delete(`${API_BASE}/commodities/reports/${filename}`);
};

export const generateCommodityReport = async (asset: 'gold' | 'silver'): Promise<void> => {
  await axios.post(`${API_BASE}/commodities/analyze`, { asset });
};

export const fetchReportContent = async (filename: string): Promise<string> => {
  const response = await axios.get(`${API_BASE}/reports/${filename}`);
  return response.data.content;
};

export const fetchDashboardOverview = async (): Promise<any> => {
  const response = await axios.get(`${API_BASE}/dashboard/overview`);
  return response.data;
};

export const fetchMarketFunds = async (query: string): Promise<any[]> => {
  const response = await axios.get(`${API_BASE}/market-funds`, { params: { query } });
  return response.data;
};

export const generateReport = async (mode: 'pre' | 'post', fundCode?: string): Promise<void> => {
  await axios.post(`${API_BASE}/generate/${mode}`, { fund_code: fundCode });
};

export interface FundItem {
  code: string;
  name: string;
  style?: string;
  focus?: string[];
  pre_market_time?: string; // HH:MM
  post_market_time?: string; // HH:MM
  is_active?: boolean;
}

export interface SettingsData {
  llm_provider: string;
  gemini_api_key_masked: string;
  openai_api_key_masked: string;
  tavily_api_key_masked: string;
}

export interface SettingsUpdate {
  llm_provider?: string;
  gemini_api_key?: string;
  openai_api_key?: string;
  tavily_api_key?: string;
}

export const fetchFunds = async (): Promise<FundItem[]> => {
  const response = await axios.get(`${API_BASE}/funds`);
  return response.data;
};

export const saveFund = async (fund: FundItem): Promise<void> => {
  await axios.put(`${API_BASE}/funds/${fund.code}`, fund);
};

export const deleteFund = async (code: string): Promise<void> => {
  await axios.delete(`${API_BASE}/funds/${code}`);
};

export interface MarketFund {
    code: string;
    name: string;
    type: string;
    pinyin: string;
}

export const searchMarketFunds = async (query: string): Promise<MarketFund[]> => {
    const response = await axios.get(`${API_BASE}/market/funds`, { params: { q: query } });
    return response.data;
};

export interface FundMarketDetails {
    info: Record<string, string>;
    performance: any[];
}

export const fetchFundMarketDetails = async (code: string): Promise<FundMarketDetails> => {
    const response = await axios.get(`${API_BASE}/market/funds/${code}/details`);
    return response.data;
};

export interface NavPoint {
    date: string;
    value: number;
}

export interface IndexData {
  name: string;
  code: string;
  price: number;
  change_pct: number;
  change_val: number;
}

export const fetchMarketIndices = async (): Promise<IndexData[]> => {
    const response = await axios.get(`${API_BASE}/market/indices`);
    return response.data;
};

export const fetchFundNavHistory = async (code: string): Promise<NavPoint[]> => {
    const response = await axios.get(`${API_BASE}/market/funds/${code}/nav`);
    return response.data;
};

export const fetchSettings = async (): Promise<SettingsData> => {
  const response = await axios.get(`${API_BASE}/settings`);
  return response.data;
};

export const saveSettings = async (settings: SettingsUpdate): Promise<void> => {
  await axios.post(`${API_BASE}/settings`, settings);
};

export interface SentimentResponse {
  report: string;
  filename: string;
}

export const runSentimentAnalysis = async (): Promise<SentimentResponse> => {
  const response = await axios.post(`${API_BASE}/sentiment/analyze`);
  return response.data;
};

export interface SentimentReportItem {
  filename: string;
  date: string;
}

export const fetchSentimentReports = async (): Promise<SentimentReportItem[]> => {
  const response = await axios.get(`${API_BASE}/sentiment/reports`);
  return response.data;
};

export const deleteSentimentReport = async (filename: string): Promise<void> => {
  await axios.delete(`${API_BASE}/sentiment/reports/${filename}`);
};
