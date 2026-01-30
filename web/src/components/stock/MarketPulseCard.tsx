import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  Paper,
  Typography,
  Skeleton,
  Tooltip,
  IconButton,
  Grid,
  Divider,
  Stack,
  alpha,
  useTheme,
  Collapse,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import RefreshIcon from '@mui/icons-material/Refresh';
import BarChartIcon from '@mui/icons-material/BarChart';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { fetchMarketSummary, fetchMarketAIBrief } from '../../api';
import type { MarketSummary, MarketAIBrief } from '../../api';

interface MarketPulseCardProps {
  onRefresh?: () => void;
}

export default function MarketPulseCard({ onRefresh }: MarketPulseCardProps) {
  const { t, i18n } = useTranslation();
  const theme = useTheme();
  const [data, setData] = useState<MarketSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // AI Brief state
  const [aiBrief, setAiBrief] = useState<MarketAIBrief | null>(null);
  const [loadingAI, setLoadingAI] = useState(false);
  const [showAI] = useState(true);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchMarketSummary();
      setData(result);
    } catch (err) {
      setError(t('stocks.market.load_error'));
      console.error('Failed to load market summary:', err);
    } finally {
      setLoading(false);
    }
  };

  const loadAIBrief = async () => {
    setLoadingAI(true);
    try {
      const result = await fetchMarketAIBrief();
      setAiBrief(result);
    } catch (err) {
      console.error('Failed to load AI brief:', err);
    } finally {
      setLoadingAI(false);
    }
  };

  useEffect(() => {
    loadData();
    loadAIBrief();
  }, []);

  const handleRefresh = () => {
    loadData();
    loadAIBrief();
    onRefresh?.();
  };

  // 格式化数字
  const formatNumber = (num: number | string | undefined): string => {
    if (num === undefined || num === null) return '-';
    const n = typeof num === 'string' ? parseFloat(num) : num;
    return isNaN(n) ? '-' : n.toLocaleString();
  };

  if (loading) {
    return (
      <Paper sx={{ p: 3, borderRadius: 3, boxShadow: '0 4px 20px rgba(0,0,0,0.05)' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Skeleton width={150} height={32} />
          <Skeleton width={32} height={32} variant="circular" />
        </Box>
        <Grid container spacing={4}>
          {[1, 2].map((i) => (
            <Grid size={{ xs: 12, md: 6 }} key={i}>
              <Skeleton height={80} />
              <Skeleton height={20} width="80%" sx={{ mt: 1 }} />
            </Grid>
          ))}
        </Grid>
      </Paper>
    );
  }

  if (error || !data) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center', borderRadius: 3 }}>
        <Typography color="error">{error || t('stocks.market.no_data')}</Typography>
        <IconButton onClick={handleRefresh} size="small" sx={{ mt: 1 }}>
          <RefreshIcon />
        </IconButton>
      </Paper>
    );
  }

  const activity = data.activity;

  // Calculate percentages for the bar
  const upCount = typeof activity?.上涨 === 'string' ? parseInt(activity.上涨) : (activity?.上涨 || 0);
  const downCount = typeof activity?.下跌 === 'string' ? parseInt(activity.下跌) : (activity?.下跌 || 0);
  const flatCount = typeof activity?.平盘 === 'string' ? parseInt(activity.平盘) : (activity?.平盘 || 0);
  const total = upCount + downCount + flatCount;
  
  const upPercent = total > 0 ? (upCount / total) * 100 : 0;
  const downPercent = total > 0 ? (downCount / total) * 100 : 0;

  const isZh = i18n.language === 'zh';

  return (
    <Paper 
      elevation={0}
      sx={{ 
        p: 3, 
        borderRadius: 3, 
        border: '1px solid',
        borderColor: 'divider',
        background: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box sx={{ p: 0.5, borderRadius: 1.5, bgcolor: alpha(theme.palette.primary.main, 0.1), color: 'primary.main' }}>
            <BarChartIcon fontSize="small" />
          </Box>
          <Typography variant="h6" sx={{ fontWeight: 700, letterSpacing: '-0.01em' }}>
            {t('stocks.market.pulse_title')}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <Box sx={{ px: 1.5, py: 0.5, borderRadius: 4, bgcolor: 'background.paper', border: '1px solid', borderColor: 'divider' }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 500, fontFamily: 'Monospace' }}>
              {data.update_time}
            </Typography>
          </Box>
          <Tooltip title={t('common.refresh')}>
            <IconButton size="small" onClick={handleRefresh} sx={{ bgcolor: 'background.paper', border: '1px solid', borderColor: 'divider' }}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Grid container spacing={4}>
        {/* Section 1: Market Breadth (上涨/下跌分布) */}
        <Grid size={{ xs: 12, md: 6 }}>
          <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1.5 }}>
              <Box>
                <Typography variant="caption" color="text.secondary" fontWeight={600} sx={{ mb: 0.5, display: 'block' }}>
                  {t('stocks.market.rising')}
                </Typography>
                <Typography variant="h5" sx={{ color: '#ef4444', fontWeight: 800, fontFamily: 'JetBrains Mono' }}>
                  {formatNumber(activity?.上涨)}
                </Typography>
              </Box>
              <Box sx={{ textAlign: 'right' }}>
                <Typography variant="caption" color="text.secondary" fontWeight={600} sx={{ mb: 0.5, display: 'block' }}>
                  {t('stocks.market.falling')}
                </Typography>
                <Typography variant="h5" sx={{ color: '#22c55e', fontWeight: 800, fontFamily: 'JetBrains Mono' }}>
                  {formatNumber(activity?.下跌)}
                </Typography>
              </Box>
            </Box>
            
            {/* Custom Progress Bar */}
            <Box sx={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', bgcolor: '#f1f5f9' }}>
              <Box sx={{ width: `${upPercent}%`, bgcolor: '#ef4444', transition: 'width 1s ease-in-out' }} />
              <Box sx={{ flex: 1, bgcolor: '#f1f5f9' }} />
              <Box sx={{ width: `${downPercent}%`, bgcolor: '#22c55e', transition: 'width 1s ease-in-out' }} />
            </Box>
            
            <Box sx={{ mt: 1, display: 'flex', justifyContent: 'center' }}>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <span>{t('stocks.market.flat')}:</span>
                <span style={{ fontWeight: 700, color: '#64748b' }}>{formatNumber(activity?.平盘)}</span>
              </Typography>
            </Box>
          </Box>
        </Grid>

        <Divider orientation="vertical" flexItem sx={{ display: { xs: 'none', md: 'block' }, mx: 2 }} />

        {/* Section 2: Limit Board (涨停/跌停) */}
        <Grid size={{ xs: 12, md: 5 }}>
          <Stack spacing={2}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box>
                <Typography variant="caption" color="text.secondary" fontWeight={600}>
                  {t('stocks.market.limit_up')}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
                  <Typography variant="h6" sx={{ color: '#ef4444', fontWeight: 800 }}>
                    {formatNumber(activity?.涨停)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    ({formatNumber(activity?.真实涨停)} {isZh ? '真实' : 'real'})
                  </Typography>
                </Box>
              </Box>
              <TrendingUpIcon sx={{ color: alpha('#ef4444', 0.2), fontSize: 32 }} />
            </Box>
            
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Box>
                <Typography variant="caption" color="text.secondary" fontWeight={600}>
                  {t('stocks.market.limit_down')}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
                  <Typography variant="h6" sx={{ color: '#22c55e', fontWeight: 800 }}>
                    {formatNumber(activity?.跌停)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    ({formatNumber(activity?.真实跌停)} {isZh ? '真实' : 'real'})
                  </Typography>
                </Box>
              </Box>
              <TrendingDownIcon sx={{ color: alpha('#22c55e', 0.2), fontSize: 32 }} />
            </Box>
          </Stack>
        </Grid>
      </Grid>

      {/* AI Brief Section */}
      <Collapse in={showAI}>
        <Divider sx={{ my: 2.5 }} />
        <Box sx={{ 
          p: 2, 
          borderRadius: 2, 
          bgcolor: alpha(theme.palette.primary.main, 0.03),
          border: '1px solid',
          borderColor: alpha(theme.palette.primary.main, 0.1),
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
            <AutoAwesomeIcon sx={{ color: 'primary.main', fontSize: 18 }} />
            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: 'primary.main' }}>
              {t('stocks.market.ai_brief_title')}
            </Typography>
            {loadingAI && (
              <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
                {t('stocks.market.ai_analyzing')}...
              </Typography>
            )}
          </Box>
          
          {loadingAI ? (
            <Box>
              <Skeleton width="100%" height={20} />
              <Skeleton width="90%" height={20} sx={{ mt: 0.5 }} />
              <Skeleton width="70%" height={20} sx={{ mt: 0.5 }} />
            </Box>
          ) : aiBrief ? (
            <Box>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: 'text.primary', 
                  lineHeight: 1.8,
                  whiteSpace: 'pre-wrap',
                }}
              >
                {aiBrief.brief}
              </Typography>
              
              {/* Industry Tags */}
              {aiBrief.top_industries && aiBrief.top_industries.length > 0 && (
                <Box sx={{ mt: 1.5, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {aiBrief.top_industries.map((ind, idx) => (
                    <Box
                      key={idx}
                      sx={{
                        px: 1.5,
                        py: 0.5,
                        borderRadius: 2,
                        bgcolor: alpha('#ef4444', 0.1),
                        border: '1px solid',
                        borderColor: alpha('#ef4444', 0.2),
                      }}
                    >
                      <Typography variant="caption" sx={{ fontWeight: 600, color: '#ef4444' }}>
                        {ind.name} ({ind.count})
                      </Typography>
                    </Box>
                  ))}
                </Box>
              )}
            </Box>
          ) : (
            <Typography variant="body2" color="text.secondary">
              {t('stocks.market.ai_unavailable')}
            </Typography>
          )}
        </Box>
      </Collapse>
    </Paper>
  );
}
