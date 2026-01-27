import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  CircularProgress,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  useTheme,
  alpha,
  Chip,
} from '@mui/material';
import {
  CalendarMonth,
  CalendarViewDay,
  CalendarViewMonth,
  DateRange,
} from '@mui/icons-material';
import { useTranslation } from 'react-i18next';
import type { ReturnsCalendarData, ReturnsCalendarEntry } from '../../api';

interface ReturnsCalendarProps {
  data: ReturnsCalendarData | null;
  loading?: boolean;
  onViewChange?: (view: 'day' | 'month' | 'year') => void;
}

interface CalendarCellProps {
  entry: ReturnsCalendarEntry;
  view: 'day' | 'month' | 'year';
}

const CalendarCell: React.FC<CalendarCellProps> = ({ entry, view }) => {
  const theme = useTheme();

  // Calculate color intensity based on percentage
  const getBackgroundColor = (pnlPct: number) => {
    const maxIntensity = 0.8;
    const absVal = Math.min(Math.abs(pnlPct), 10); // Cap at 10% for color scaling
    const intensity = (absVal / 10) * maxIntensity;

    if (pnlPct > 0) {
      return alpha(theme.palette.success.main, 0.1 + intensity);
    } else if (pnlPct < 0) {
      return alpha(theme.palette.error.main, 0.1 + intensity);
    }
    return alpha(theme.palette.grey[500], 0.1);
  };

  const getTextColor = (pnlPct: number) => {
    if (pnlPct > 0) return theme.palette.success.dark;
    if (pnlPct < 0) return theme.palette.error.dark;
    return theme.palette.text.secondary;
  };

  const formatDate = (date: string) => {
    if (view === 'year') return date; // YYYY
    if (view === 'month') {
      const parts = date.split('-');
      return `${parts[1]}月`;
    }
    // day: show day of month
    const day = date.split('-')[2];
    return parseInt(day, 10).toString();
  };

  const getTooltipContent = () => {
    const pnlStr = entry.pnl >= 0 ? `+¥${entry.pnl.toFixed(2)}` : `¥${entry.pnl.toFixed(2)}`;
    const pctStr = entry.pnl_pct >= 0 ? `+${entry.pnl_pct.toFixed(2)}%` : `${entry.pnl_pct.toFixed(2)}%`;
    return `${entry.date}\n${pnlStr} (${pctStr})`;
  };

  return (
    <Tooltip title={getTooltipContent()} placement="top">
      <Box
        sx={{
          flex: view === 'day' ? '0 0 calc(14.28% - 4px)' : view === 'month' ? '0 0 calc(25% - 6px)' : '1 1 auto',
          minWidth: view === 'day' ? 36 : view === 'month' ? 80 : 100,
          height: view === 'day' ? 36 : view === 'month' ? 56 : 70,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: 1,
          bgcolor: getBackgroundColor(entry.pnl_pct),
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          '&:hover': {
            transform: 'scale(1.05)',
            boxShadow: `0 2px 8px ${alpha(theme.palette.common.black, 0.15)}`,
          },
        }}
      >
        <Typography
          sx={{
            fontSize: view === 'day' ? '0.75rem' : '0.875rem',
            fontWeight: 600,
            color: getTextColor(entry.pnl_pct),
            fontFamily: 'JetBrains Mono, monospace',
          }}
        >
          {formatDate(entry.date)}
        </Typography>
        {view !== 'day' && (
          <Typography
            sx={{
              fontSize: view === 'month' ? '0.7rem' : '0.75rem',
              color: getTextColor(entry.pnl_pct),
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            {entry.pnl_pct >= 0 ? '+' : ''}{entry.pnl_pct.toFixed(1)}%
          </Typography>
        )}
      </Box>
    </Tooltip>
  );
};

const ReturnsCalendar: React.FC<ReturnsCalendarProps> = ({
  data,
  loading = false,
  onViewChange,
}) => {
  const theme = useTheme();
  const { t } = useTranslation();
  const [view, setView] = useState<'day' | 'month' | 'year'>('day');

  const handleViewChange = (_: React.MouseEvent, newView: 'day' | 'month' | 'year' | null) => {
    if (newView) {
      setView(newView);
      onViewChange?.(newView);
    }
  };

  // Generate calendar for current month, merging with actual data
  const groupedData = useMemo(() => {
    if (view !== 'day') return null;

    // Create a map of existing data by date
    const dataMap = new Map<string, ReturnsCalendarEntry>();
    if (data?.data) {
      data.data.forEach(entry => {
        dataMap.set(entry.date, entry);
      });
    }

    // Get current month's dates
    const now = new Date();
    const year = now.getFullYear();
    const month = now.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);

    const weeks: ReturnsCalendarEntry[][] = [];
    let currentWeek: ReturnsCalendarEntry[] = [];

    // Pad first week with empty cells
    const firstDayOfWeek = firstDay.getDay();
    for (let i = 0; i < firstDayOfWeek; i++) {
      currentWeek.push({ date: '', pnl: 0, pnl_pct: 0, is_trading_day: false });
    }

    // Generate all days of the month
    for (let day = 1; day <= lastDay.getDate(); day++) {
      const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
      const existingEntry = dataMap.get(dateStr);

      const entry: ReturnsCalendarEntry = existingEntry || {
        date: dateStr,
        pnl: 0,
        pnl_pct: 0,
        is_trading_day: true,
      };

      currentWeek.push(entry);

      // Start new week after Saturday
      if (currentWeek.length === 7) {
        weeks.push(currentWeek);
        currentWeek = [];
      }
    }

    // Push remaining days
    if (currentWeek.length > 0) {
      weeks.push(currentWeek);
    }

    return weeks;
  }, [data, view]);

  // Generate month data for current year
  const monthData = useMemo(() => {
    if (view !== 'month') return null;

    const dataMap = new Map<string, ReturnsCalendarEntry>();
    if (data?.data) {
      data.data.forEach(entry => {
        dataMap.set(entry.date, entry);
      });
    }

    const now = new Date();
    const year = now.getFullYear();
    const months: ReturnsCalendarEntry[] = [];

    for (let month = 1; month <= 12; month++) {
      const dateStr = `${year}-${String(month).padStart(2, '0')}`;
      const existingEntry = dataMap.get(dateStr);

      months.push(existingEntry || {
        date: dateStr,
        pnl: 0,
        pnl_pct: 0,
        is_trading_day: true,
      });
    }

    return months;
  }, [data, view]);

  // Generate year data (last 5 years)
  const yearData = useMemo(() => {
    if (view !== 'year') return null;

    const dataMap = new Map<string, ReturnsCalendarEntry>();
    if (data?.data) {
      data.data.forEach(entry => {
        dataMap.set(entry.date, entry);
      });
    }

    const now = new Date();
    const currentYear = now.getFullYear();
    const years: ReturnsCalendarEntry[] = [];

    for (let year = currentYear - 4; year <= currentYear; year++) {
      const dateStr = `${year}`;
      const existingEntry = dataMap.get(dateStr);

      years.push(existingEntry || {
        date: dateStr,
        pnl: 0,
        pnl_pct: 0,
        is_trading_day: true,
      });
    }

    return years;
  }, [data, view]);

  if (loading) {
    return (
      <Paper
        elevation={0}
        sx={{
          p: 3,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 300,
          background: alpha(theme.palette.background.paper, 0.8),
          border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
          borderRadius: 3,
        }}
      >
        <Box sx={{ textAlign: 'center' }}>
          <CircularProgress size={32} sx={{ mb: 1 }} />
          <Typography color="text.secondary">
            {t('portfolio.loadingCalendar', '加载日历数据...')}
          </Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        background: alpha(theme.palette.background.paper, 0.8),
        border: `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        borderRadius: 3,
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <CalendarMonth sx={{ fontSize: 20, color: 'primary.main' }} />
          <Typography variant="subtitle1" fontWeight={600}>
            {t('portfolio.returnsCalendar', '收益日历')}
          </Typography>
        </Box>

        <ToggleButtonGroup
          size="small"
          value={view}
          exclusive
          onChange={handleViewChange}
          sx={{
            '& .MuiToggleButton-root': {
              px: 1.5,
              py: 0.5,
              textTransform: 'none',
              fontSize: '0.75rem',
            },
          }}
        >
          <ToggleButton value="day">
            <CalendarViewDay sx={{ fontSize: 16, mr: 0.5 }} />
            {t('portfolio.viewDay', '日')}
          </ToggleButton>
          <ToggleButton value="month">
            <CalendarViewMonth sx={{ fontSize: 16, mr: 0.5 }} />
            {t('portfolio.viewMonth', '月')}
          </ToggleButton>
          <ToggleButton value="year">
            <DateRange sx={{ fontSize: 16, mr: 0.5 }} />
            {t('portfolio.viewYear', '年')}
          </ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Stats Summary */}
      {data?.stats && (
        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Chip
            size="small"
            label={t('portfolio.profitablePeriods', '盈利 {{count}}', { count: data.stats.profitable_periods })}
            sx={{ bgcolor: alpha(theme.palette.success.main, 0.1), color: 'success.main' }}
          />
          <Chip
            size="small"
            label={t('portfolio.lossPeriods', '亏损 {{count}}', { count: data.stats.loss_periods })}
            sx={{ bgcolor: alpha(theme.palette.error.main, 0.1), color: 'error.main' }}
          />
          {data.stats.best_period && (
            <Chip
              size="small"
              label={t('portfolio.bestPeriod', '最佳: {{date}} +{{pct}}%', {
                date: data.stats.best_period.date,
                pct: data.stats.best_period.pnl_pct.toFixed(1),
              })}
              sx={{ bgcolor: alpha(theme.palette.success.main, 0.1), color: 'success.dark' }}
            />
          )}
        </Box>
      )}

      {/* Calendar Content */}
      {view === 'day' && groupedData ? (
        // Day View - Calendar Grid
        <Box sx={{ width: '100%' }}>
          {/* Weekday headers */}
          <Box sx={{ display: 'flex', gap: 0.5, mb: 0.5 }}>
            {['日', '一', '二', '三', '四', '五', '六'].map((day) => (
              <Box
                key={day}
                sx={{
                  flex: '0 0 calc(14.28% - 4px)',
                  textAlign: 'center',
                  fontSize: '0.75rem',
                  fontWeight: 600,
                  color: 'text.secondary',
                  py: 0.5,
                }}
              >
                {day}
              </Box>
            ))}
          </Box>
          {/* Weeks */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            {groupedData.map((week, weekIdx) => (
              <Box key={weekIdx} sx={{ display: 'flex', gap: 0.5 }}>
                {week.map((entry, dayIdx) => (
                  entry.date ? (
                    <CalendarCell key={entry.date} entry={entry} view="day" />
                  ) : (
                    <Box key={`empty-${dayIdx}`} sx={{ flex: '0 0 calc(14.28% - 4px)', height: 36 }} />
                  )
                ))}
              </Box>
            ))}
          </Box>
        </Box>
      ) : view === 'month' && monthData ? (
        // Month View - 4x3 Grid
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 1,
            width: '100%',
          }}
        >
          {monthData.map((entry) => (
            <CalendarCell key={entry.date} entry={entry} view="month" />
          ))}
        </Box>
      ) : view === 'year' && yearData ? (
        // Year View - Horizontal row
        <Box
          sx={{
            display: 'flex',
            gap: 1,
            width: '100%',
          }}
        >
          {yearData.map((entry) => (
            <CalendarCell key={entry.date} entry={entry} view="year" />
          ))}
        </Box>
      ) : (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <Typography color="text.secondary">
            {t('portfolio.noCalendarData', '暂无日历数据')}
          </Typography>
        </Box>
      )}

      {/* Legend */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', mt: 2, gap: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: 0.5,
              bgcolor: alpha(theme.palette.error.main, 0.6),
            }}
          />
          <Typography variant="caption" color="text.secondary">
            {t('portfolio.loss', '亏损')}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: 0.5,
              bgcolor: alpha(theme.palette.grey[500], 0.2),
            }}
          />
          <Typography variant="caption" color="text.secondary">
            {t('portfolio.neutral', '持平')}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: 0.5,
              bgcolor: alpha(theme.palette.success.main, 0.6),
            }}
          />
          <Typography variant="caption" color="text.secondary">
            {t('portfolio.profit', '盈利')}
          </Typography>
        </Box>
      </Box>
    </Paper>
  );
};

export default ReturnsCalendar;
