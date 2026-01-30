import React from 'react';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  Tooltip,
  useTheme,
  alpha,
  Chip,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import {
  TrendingUp,
  CalendarToday,
  ShowChart,
  Speed,
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import type { ReturnsSummary } from '../../api';

interface ReturnsOverviewProps {
  data: ReturnsSummary | null;
  loading?: boolean;
}

interface MetricCardProps {
  label: string;
  value: number | null | undefined;
  suffix?: string;
  prefix?: string;
  showSign?: boolean;
  icon: React.ReactNode;
  tooltip?: string;
  secondary?: string;
  trend?: 'up' | 'down' | 'neutral';
}

const formatSignedCurrency = (value: number | null | undefined) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `¥${value >= 0 ? '+' : ''}${value.toFixed(2)}`;
};

const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  suffix = '%',
  prefix = '',
  showSign = true,
  icon,
  tooltip,
  secondary,
  trend,
}) => {
  const theme = useTheme();

  const isMissing = value === null || value === undefined || Number.isNaN(value);

  const getColor = () => {
    if (isMissing) return theme.palette.text.primary;
    if (trend === 'up' || (showSign && (value as number) > 0)) return theme.palette.success.main;
    if (trend === 'down' || (showSign && (value as number) < 0)) return theme.palette.error.main;
    return theme.palette.text.primary;
  };

  const content = (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        height: '100%',
        background: alpha(theme.palette.background.paper, 0.8),
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        borderRadius: 2,
        transition: 'all 0.2s ease',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.1)}`,
        },
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Box
          sx={{
            p: 0.5,
            borderRadius: 1,
            bgcolor: alpha(getColor(), 0.1),
            color: getColor(),
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          {icon}
        </Box>
        <Typography variant="caption" color="text.secondary" fontWeight={500}>
          {label}
        </Typography>
      </Box>
      <Box>
        <Typography
          variant="h5"
          fontWeight={700}
          sx={{
            color: getColor(),
            fontFamily: 'JetBrains Mono, monospace',
          }}
        >
          {isMissing
            ? '—'
            : (
              <>
                {showSign && (value as number) > 0 ? '+' : ''}
                {prefix}
                {(value as number).toFixed(suffix === '%' ? 2 : 0)}
                {suffix}
              </>
            )}
        </Typography>
        {secondary && (
          <Typography variant="caption" color="text.secondary">
            {secondary}
          </Typography>
        )}
      </Box>
    </Paper>
  );

  return tooltip ? (
    <Tooltip title={tooltip} placement="top">
      {content}
    </Tooltip>
  ) : content;
};

const ReturnsOverview: React.FC<ReturnsOverviewProps> = ({ data, loading = false }) => {
  const theme = useTheme();
  const { t } = useTranslation();

  if (loading || !data) {
    return (
      <Paper
        elevation={0}
        sx={{
          p: 3,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 150,
          background: alpha(theme.palette.background.paper, 0.8),
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          borderRadius: 3,
        }}
      >
        <Box sx={{ textAlign: 'center' }}>
          <CircularProgress size={32} sx={{ mb: 1 }} />
          <Typography color="text.secondary">
            {t('portfolio.loadingReturns', '加载收益数据...')}
          </Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Box sx={{ mb: 3 }}>
      {/* Title */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
        <ShowChart sx={{ fontSize: 20, color: 'primary.main' }} />
        <Typography variant="subtitle1" fontWeight={600}>
          {t('portfolio.returnsSummary', '收益总览')}
        </Typography>
        {data.total_trading_days > 0 && (
          <Chip
            size="small"
            label={t('portfolio.tradingDaysCount', '{{days}}个交易日', { days: data.total_trading_days })}
            sx={{ fontSize: '0.7rem' }}
          />
        )}
      </Box>

      {/* Metric Cards Grid */}
      <Grid container spacing={2}>
        {/* Today's Return */}
        <Grid size={{ xs: 6, sm: 4, md: 2 }}>
          <MetricCard
            label={t('portfolio.todayPnl', '今日收益')}
            value={data.today_pnl_pct}
            icon={<CalendarToday sx={{ fontSize: 16 }} />}
            secondary={formatSignedCurrency(data.today_pnl)}
          />
        </Grid>

        {/* Week Return */}
        <Grid size={{ xs: 6, sm: 4, md: 2 }}>
          <MetricCard
            label={t('portfolio.weekPnl', '本周收益')}
            value={data.week_pnl_pct}
            icon={<TrendingUp sx={{ fontSize: 16 }} />}
            secondary={formatSignedCurrency(data.week_pnl)}
          />
        </Grid>

        {/* Month Return */}
        <Grid size={{ xs: 6, sm: 4, md: 2 }}>
          <MetricCard
            label={t('portfolio.monthPnl', '本月收益')}
            value={data.month_pnl_pct}
            icon={<TrendingUp sx={{ fontSize: 16 }} />}
            secondary={formatSignedCurrency(data.month_pnl)}
          />
        </Grid>

        {/* Total Return */}
        <Grid size={{ xs: 6, sm: 4, md: 2 }}>
          <MetricCard
            label={t('portfolio.totalReturn', '累计收益')}
            value={data.total_pnl_pct}
            icon={<ShowChart sx={{ fontSize: 16 }} />}
            secondary={formatSignedCurrency(data.total_pnl)}
          />
        </Grid>

        {/* Annualized Return */}
        <Grid size={{ xs: 6, sm: 4, md: 2 }}>
          <MetricCard
            label={t('portfolio.annualizedReturn', '年化收益')}
            value={data.annualized_return}
            icon={<Speed sx={{ fontSize: 16 }} />}
            tooltip={t('portfolio.annualizedReturnTooltip', '基于历史数据计算的年化收益率')}
          />
        </Grid>
      </Grid>
    </Box>
  );
};

export default ReturnsOverview;
