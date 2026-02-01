import { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  Typography,
  CircularProgress,
  Paper,
  Grid,
  Chip,
  Collapse,
  IconButton,
  Button
} from '@mui/material';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import SpeedIcon from '@mui/icons-material/Speed';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import RefreshIcon from '@mui/icons-material/Refresh';
import { fetchStockQuant, fetchQuantAIInterpretation } from '../../api';
import type { QuantData, QuantInterpretationResponse } from '../../api';

interface QuantTabProps {
  code: string;
}

export default function QuantTab({ code }: QuantTabProps) {
  const { t } = useTranslation();
  const [data, setData] = useState<QuantData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // AI Interpretation state
  const [aiExpanded, setAiExpanded] = useState(false);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiData, setAiData] = useState<QuantInterpretationResponse | null>(null);
  const [aiError, setAiError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchStockQuant(code);
        setData(result);
      } catch (err) {
        setError(t('stocks.professional.load_error'));
        console.error('Failed to load quant data:', err);
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [code, t]);

  // Load AI interpretation when expanded
  const loadAIInterpretation = async () => {
    if (aiData) return; // Already loaded
    setAiLoading(true);
    setAiError(null);
    try {
      const result = await fetchQuantAIInterpretation(code);
      setAiData(result);
    } catch (err) {
      setAiError(t('stocks.ai.interpretation_error'));
      console.error('Failed to load AI interpretation:', err);
    } finally {
      setAiLoading(false);
    }
  };

  const handleAiToggle = () => {
    const newExpanded = !aiExpanded;
    setAiExpanded(newExpanded);
    if (newExpanded && !aiData && !aiLoading) {
      loadAIInterpretation();
    }
  };

  const handleAiRefresh = () => {
    setAiData(null);
    loadAIInterpretation();
  };

  if (loading) {
    return (
      <Box sx={{ py: 8, textAlign: 'center' }}>
        <CircularProgress size={32} sx={{ color: '#6366f1' }} />
        <Typography sx={{ mt: 2, color: '#64748b' }}>{t('common.loading')}</Typography>
      </Box>
    );
  }

  if (error || !data) {
    return (
      <Box sx={{ py: 8, textAlign: 'center' }}>
        <Typography color="error">{error || t('stocks.professional.no_data')}</Typography>
      </Box>
    );
  }

  const getSignalColor = (signal?: string) => {
    switch (signal) {
      case 'bullish': return '#22c55e';
      case 'bearish': return '#ef4444';
      case 'overbought': return '#f59e0b';
      case 'oversold': return '#8b5cf6';
      default: return '#64748b';
    }
  };

  const getSignalBg = (signal?: string) => {
    const color = getSignalColor(signal);
    return color + '20';
  };

  const getSignalLabel = (signal?: string) => {
    switch (signal) {
      case 'bullish': return t('stocks.quant.bullish');
      case 'bearish': return t('stocks.quant.bearish');
      case 'overbought': return t('stocks.quant.overbought');
      case 'oversold': return t('stocks.quant.oversold');
      default: return t('stocks.quant.neutral');
    }
  };

  const formatValue = (val: number | undefined | null) => {
    if (val == null) return '---';
    return val.toFixed(2);
  };

  const renderSignalIndicator = (
    name: string,
    signal: { signal: string; value?: number | { upper: number; mid?: number; lower: number; close: number } }
  ) => {
    const color = getSignalColor(signal.signal);
    const displayValue = typeof signal.value === 'number'
      ? formatValue(signal.value)
      : signal.value
        ? `${formatValue(signal.value.close)}`
        : '---';

    return (
      <Paper elevation={0} sx={{ p: 2, bgcolor: '#fff', borderRadius: '10px', border: '1px solid #f1f5f9' }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="overline" sx={{ color: '#64748b', fontWeight: 800, fontSize: '0.65rem' }}>
              {name}
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 800, fontFamily: 'JetBrains Mono' }}>
              {displayValue}
            </Typography>
          </Box>
          <Chip
            label={getSignalLabel(signal.signal)}
            size="small"
            sx={{
              bgcolor: getSignalBg(signal.signal),
              color: color,
              fontWeight: 800,
              fontSize: '0.7rem'
            }}
          />
        </Box>
      </Paper>
    );
  };

  const getOverallSignalColor = () => {
    if (!data.overall_signal) return '#64748b';
    if (data.overall_signal.direction === 'bullish') return '#22c55e';
    if (data.overall_signal.direction === 'bearish') return '#ef4444';
    return '#64748b';
  };

  const renderGauge = () => {
    if (!data.overall_signal) return null;

    const score = data.overall_signal.score;
    const normalizedScore = (score + 4) / 8; // Normalize -4 to 4 into 0 to 1
    const angle = -90 + (normalizedScore * 180); // Map to -90 to 90 degrees

    return (
      <Box sx={{ position: 'relative', width: 120, height: 70, mx: 'auto' }}>
        <svg viewBox="0 0 120 70" style={{ width: '100%', height: '100%' }}>
          {/* Background arc */}
          <path
            d="M 10 60 A 50 50 0 0 1 110 60"
            fill="none"
            stroke="#e2e8f0"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Colored segments */}
          <path
            d="M 10 60 A 50 50 0 0 1 35 20"
            fill="none"
            stroke="#ef4444"
            strokeWidth="8"
            strokeLinecap="round"
          />
          <path
            d="M 35 20 A 50 50 0 0 1 60 10"
            fill="none"
            stroke="#f59e0b"
            strokeWidth="8"
            strokeLinecap="round"
          />
          <path
            d="M 60 10 A 50 50 0 0 1 85 20"
            fill="none"
            stroke="#22c55e"
            strokeWidth="8"
            strokeLinecap="round"
          />
          <path
            d="M 85 20 A 50 50 0 0 1 110 60"
            fill="none"
            stroke="#16a34a"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Needle */}
          <g transform={`rotate(${angle}, 60, 60)`}>
            <line
              x1="60"
              y1="60"
              x2="60"
              y2="20"
              stroke={getOverallSignalColor()}
              strokeWidth="3"
              strokeLinecap="round"
            />
            <circle cx="60" cy="60" r="6" fill={getOverallSignalColor()} />
          </g>
        </svg>
      </Box>
    );
  };

  // Prepare MACD chart data
  const macdData = (data.factors ?? []).slice().reverse().slice(-20).map(f => ({
    date: f.trade_date?.substring(4) || '',
    macd: f.macd || 0
  }));

  const renderMacdChart = () => {
    if (macdData.length < 2) return null;
    const width = 300;
    const height = 80;
    const padding = 10;
    const barWidth = (width - 2 * padding) / macdData.length - 2;

    const maxAbs = Math.max(...macdData.map(d => Math.abs(d.macd)));
    const scale = (height / 2 - padding) / (maxAbs || 1);

    return (
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 80 }}>
        {/* Center line */}
        <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="#e2e8f0" strokeWidth="1" />
        {/* Bars */}
        {macdData.map((d, i) => {
          const x = padding + i * (barWidth + 2);
          const barHeight = Math.abs(d.macd) * scale;
          const y = d.macd >= 0 ? height / 2 - barHeight : height / 2;
          const color = d.macd >= 0 ? '#ef4444' : '#22c55e';
          return (
            <rect key={i} x={x} y={y} width={barWidth} height={barHeight || 1} fill={color} rx="1" />
          );
        })}
      </svg>
    );
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* Overall Signal Card */}
      {data.overall_signal && (
        <Paper elevation={0} sx={{ p: 3, bgcolor: '#f8fafc', borderRadius: '12px', border: '1px solid #e2e8f0' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box sx={{
                width: 48, height: 48, borderRadius: '12px',
                bgcolor: getOverallSignalColor() + '20',
                display: 'flex', alignItems: 'center', justifyContent: 'center'
              }}>
                <SpeedIcon sx={{ color: getOverallSignalColor(), fontSize: 24 }} />
              </Box>
              <Box>
                <Typography variant="overline" sx={{ color: '#64748b', fontWeight: 800 }}>
                  {t('stocks.quant.overall_signal')}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {data.overall_signal.direction === 'bullish' ? (
                    <TrendingUpIcon sx={{ color: '#22c55e', fontSize: 24 }} />
                  ) : data.overall_signal.direction === 'bearish' ? (
                    <TrendingDownIcon sx={{ color: '#ef4444', fontSize: 24 }} />
                  ) : null}
                  <Typography variant="h5" sx={{ fontWeight: 900, color: getOverallSignalColor(), textTransform: 'uppercase' }}>
                    {t(`stocks.quant.${data.overall_signal.direction}`)}
                  </Typography>
                </Box>
              </Box>
            </Box>
            {renderGauge()}
          </Box>
          <Typography variant="caption" sx={{ color: '#94a3b8', mt: 1, display: 'block' }}>
            {t('stocks.quant.strength')}: {t(`stocks.quant.strength_${data.overall_signal.strength}`)} ({data.overall_signal.score > 0 ? '+' : ''}{data.overall_signal.score})
          </Typography>
        </Paper>
      )}

      {/* Technical Indicators Grid */}
      <Box>
        <Typography variant="overline" sx={{ color: '#0f172a', fontWeight: 900, mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
          <ShowChartIcon sx={{ fontSize: 18, color: '#6366f1' }} />
          {t('stocks.quant.tech_indicators')}
        </Typography>
        <Grid container spacing={2}>
          <Grid size={6}>
            {renderSignalIndicator('MACD', data.signals.macd)}
          </Grid>
          <Grid size={6}>
            {renderSignalIndicator('KDJ', data.signals.kdj)}
          </Grid>
          <Grid size={6}>
            {renderSignalIndicator('RSI', data.signals.rsi)}
          </Grid>
          <Grid size={6}>
            {renderSignalIndicator('BOLL', data.signals.boll)}
          </Grid>
        </Grid>
      </Box>

      {/* MACD Histogram */}
      {macdData.length > 1 && (
        <Box>
          <Typography variant="overline" sx={{ color: '#0f172a', fontWeight: 900, mb: 2, display: 'block' }}>
            {t('stocks.quant.macd_histogram')}
          </Typography>
          <Paper elevation={0} sx={{ p: 2, bgcolor: '#fcfcfc', borderRadius: '10px', border: '1px solid #f1f5f9' }}>
            {renderMacdChart()}
          </Paper>
        </Box>
      )}

      {/* Chip Distribution Summary */}
      {data.chip_summary && (
        <Box>
          <Typography variant="overline" sx={{ color: '#0f172a', fontWeight: 900, mb: 2, display: 'block' }}>
            {t('stocks.quant.chip_distribution')}
          </Typography>
          <Grid container spacing={2}>
            <Grid size={6}>
              <Paper elevation={0} sx={{ p: 2, bgcolor: '#fff', borderRadius: '10px', border: '1px solid #f1f5f9' }}>
                <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 700 }}>
                  {t('stocks.quant.winner_rate')}
                </Typography>
                <Typography variant="h6" sx={{
                  fontWeight: 800,
                  fontFamily: 'JetBrains Mono',
                  color: (data.chip_summary.winner_rate || 0) > 50 ? '#22c55e' : '#ef4444'
                }}>
                  {data.chip_summary.winner_rate != null ? `${data.chip_summary.winner_rate.toFixed(1)}%` : '---'}
                </Typography>
              </Paper>
            </Grid>
            <Grid size={6}>
              <Paper elevation={0} sx={{ p: 2, bgcolor: '#fff', borderRadius: '10px', border: '1px solid #f1f5f9' }}>
                <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 700 }}>
                  {t('stocks.quant.avg_cost')}
                </Typography>
                <Typography variant="h6" sx={{ fontWeight: 800, fontFamily: 'JetBrains Mono' }}>
                  {formatValue(data.chip_summary.weight_avg)}
                </Typography>
              </Paper>
            </Grid>
            <Grid size={4}>
              <Paper elevation={0} sx={{ p: 1.5, bgcolor: '#f8fafc', borderRadius: '8px' }}>
                <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 700, fontSize: '0.65rem' }}>
                  5% {t('stocks.quant.cost')}
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 700, fontFamily: 'JetBrains Mono' }}>
                  {formatValue(data.chip_summary.cost_5pct)}
                </Typography>
              </Paper>
            </Grid>
            <Grid size={4}>
              <Paper elevation={0} sx={{ p: 1.5, bgcolor: '#f8fafc', borderRadius: '8px' }}>
                <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 700, fontSize: '0.65rem' }}>
                  50% {t('stocks.quant.cost')}
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 700, fontFamily: 'JetBrains Mono' }}>
                  {formatValue(data.chip_summary.cost_50pct)}
                </Typography>
              </Paper>
            </Grid>
            <Grid size={4}>
              <Paper elevation={0} sx={{ p: 1.5, bgcolor: '#f8fafc', borderRadius: '8px' }}>
                <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 700, fontSize: '0.65rem' }}>
                  95% {t('stocks.quant.cost')}
                </Typography>
                <Typography variant="body2" sx={{ fontWeight: 700, fontFamily: 'JetBrains Mono' }}>
                  {formatValue(data.chip_summary.cost_95pct)}
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </Box>
      )}

      {/* AI Signal Interpretation Section */}
      <Box>
        <Paper
          elevation={0}
          sx={{
            borderRadius: '12px',
            border: '1px solid #e2e8f0',
            overflow: 'hidden',
            bgcolor: aiExpanded ? '#faf5ff' : '#f8fafc'
          }}
        >
          {/* Header - Always visible */}
          <Box
            sx={{
              p: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              cursor: 'pointer',
              '&:hover': { bgcolor: '#f1f5f9' }
            }}
            onClick={handleAiToggle}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
              <Box sx={{
                width: 36,
                height: 36,
                borderRadius: '10px',
                bgcolor: '#8b5cf6',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <AutoAwesomeIcon sx={{ color: '#fff', fontSize: 20 }} />
              </Box>
              <Box>
                <Typography variant="subtitle2" sx={{ fontWeight: 800, color: '#1e293b' }}>
                  {t('stocks.ai.signal_interpretation')}
                </Typography>
                <Typography variant="caption" sx={{ color: '#64748b' }}>
                  {t('stocks.ai.signal_interpretation_hint')}
                </Typography>
              </Box>
            </Box>
            <IconButton size="small" sx={{ color: '#64748b' }}>
              {aiExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>

          {/* Expandable Content */}
          <Collapse in={aiExpanded}>
            <Box sx={{ p: 2, pt: 0, borderTop: '1px solid #e2e8f0' }}>
              {aiLoading ? (
                <Box sx={{ py: 4, textAlign: 'center' }}>
                  <CircularProgress size={28} sx={{ color: '#8b5cf6' }} />
                  <Typography variant="caption" sx={{ color: '#64748b', display: 'block', mt: 1 }}>
                    {t('stocks.ai.analyzing')}
                  </Typography>
                </Box>
              ) : aiError ? (
                <Box sx={{ py: 3, textAlign: 'center' }}>
                  <Typography color="error" variant="body2">{aiError}</Typography>
                  <Button
                    size="small"
                    startIcon={<RefreshIcon />}
                    onClick={handleAiRefresh}
                    sx={{ mt: 1, color: '#8b5cf6' }}
                  >
                    {t('common.refresh')}
                  </Button>
                </Box>
              ) : aiData ? (
                <Box sx={{ pt: 2 }}>
                  {/* Pattern */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="overline" sx={{ color: '#7c3aed', fontWeight: 800 }}>
                      {t('stocks.ai.current_pattern')}
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 800, color: '#1e293b' }}>
                      {aiData.interpretation.pattern}
                    </Typography>
                  </Box>

                  {/* Interpretation */}
                  <Box sx={{
                    p: 2,
                    bgcolor: '#fff',
                    borderRadius: '10px',
                    border: '1px solid #e9d5ff',
                    mb: 2
                  }}>
                    <Typography variant="body2" sx={{ color: '#334155', lineHeight: 1.7 }}>
                      {aiData.interpretation.interpretation}
                    </Typography>
                  </Box>

                  {/* Action Advice */}
                  <Box sx={{
                    p: 2,
                    bgcolor: '#f0fdf4',
                    borderRadius: '10px',
                    border: '1px solid #bbf7d0'
                  }}>
                    <Typography variant="overline" sx={{ color: '#16a34a', fontWeight: 800, mb: 0.5, display: 'block' }}>
                      {t('stocks.ai.action_advice')}
                    </Typography>
                    <Typography variant="body2" sx={{ color: '#166534', fontWeight: 600 }}>
                      {aiData.interpretation.action}
                    </Typography>
                  </Box>

                  {/* Refresh button */}
                  <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                    <Button
                      size="small"
                      startIcon={<RefreshIcon />}
                      onClick={handleAiRefresh}
                      sx={{ color: '#64748b', fontSize: '0.75rem' }}
                    >
                      {t('common.refresh')}
                    </Button>
                  </Box>
                </Box>
              ) : null}
            </Box>
          </Collapse>
        </Paper>
      </Box>
    </Box>
  );
}
