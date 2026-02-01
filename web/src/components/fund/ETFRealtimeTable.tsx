import { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Chip,
  Skeleton,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

import { fetchETFRealtime,  } from '../../api';
import type{ ETFRealtimeResponse, ETFRealtimeItem } from '../../api';
interface Props {
  limit?: number;
  onETFClick?: (code: string, name: string) => void;
}

export default function ETFRealtimeTable({ limit = 30, onETFClick }: Props) {
  const [data, setData] = useState<ETFRealtimeResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>('');

  const loadData = async () => {
    setLoading(true);
    try {
      const result = await fetchETFRealtime(limit);
      setData(result);
      setLastUpdate(new Date().toLocaleTimeString('zh-CN'));
    } catch (err: any) {
      console.error('Failed to load ETF realtime:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    // Auto refresh every 30 seconds during trading hours
    const interval = setInterval(() => {
      const now = new Date();
      const hours = now.getHours();
      const minutes = now.getMinutes();
      const time = hours * 60 + minutes;
      // Trading hours: 9:30-11:30, 13:00-15:00
      const isTradingHours =
        (time >= 570 && time <= 690) || (time >= 780 && time <= 900);
      if (isTradingHours) {
        loadData();
      }
    }, 30000);

    return () => clearInterval(interval);
  }, [limit]);

  const formatAmount = (amount: number) => {
    if (amount >= 100000000) {
      return `${(amount / 100000000).toFixed(2)}亿`;
    } else if (amount >= 10000) {
      return `${(amount / 10000).toFixed(2)}万`;
    }
    return amount.toFixed(2);
  };

  const formatVolume = (volume: number) => {
    if (volume >= 10000) {
      return `${(volume / 10000).toFixed(2)}万手`;
    }
    return `${volume}手`;
  };

  const renderChangeValue = (value: number, showIcon: boolean = true) => {
    const isPositive = value > 0;
    const color = isPositive ? '#22c55e' : value < 0 ? '#ef4444' : '#64748b';
    const Icon = isPositive ? TrendingUpIcon : TrendingDownIcon;

    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
        {showIcon && value !== 0 && <Icon sx={{ fontSize: 14, color }} />}
        <Typography
          sx={{
            color,
            fontWeight: 600,
            fontSize: '0.8rem',
            fontFamily: 'JetBrains Mono',
          }}
        >
          {isPositive ? '+' : ''}{value.toFixed(2)}%
        </Typography>
      </Box>
    );
  };

  return (
    <Paper
      elevation={0}
      sx={{
        borderRadius: '16px',
        border: '1px solid #f1f5f9',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <Box
        sx={{
          p: 2,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          borderBottom: '1px solid #f1f5f9',
          bgcolor: '#fafafa',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          <ShowChartIcon sx={{ color: '#3b82f6' }} />
          <Typography sx={{ fontWeight: 700, color: '#1e293b' }}>
            ETF 实时行情
          </Typography>
          <Chip
            label="按成交额排序"
            size="small"
            sx={{
              fontSize: '0.6rem',
              height: 18,
              bgcolor: '#eff6ff',
              color: '#3b82f6',
            }}
          />
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
          {lastUpdate && (
            <Tooltip title="最后更新时间">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <AccessTimeIcon sx={{ fontSize: 14, color: '#94a3b8' }} />
                <Typography sx={{ fontSize: '0.7rem', color: '#94a3b8' }}>
                  {lastUpdate}
                </Typography>
              </Box>
            </Tooltip>
          )}
          <Tooltip title="刷新">
            <IconButton
              onClick={loadData}
              size="small"
              sx={{ color: '#94a3b8', '&:hover': { color: '#6366f1' } }}
            >
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Table */}
      {loading ? (
        <Box sx={{ p: 2 }}>
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} variant="rectangular" height={48} sx={{ mb: 1, borderRadius: 1 }} />
          ))}
        </Box>
      ) : (
        <TableContainer sx={{ maxHeight: 500 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  ETF名称
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  最新价
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  涨跌幅
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  成交额
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  成交量
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  换手率
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  振幅
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data?.etfs.map((etf, idx) => {
                const amplitude = etf.high && etf.low && etf.prev_close
                  ? ((etf.high - etf.low) / etf.prev_close * 100)
                  : 0;
                
                return (
                  <TableRow
                    key={etf.code}
                    hover
                    onClick={() => onETFClick?.(etf.code, etf.name)}
                    sx={{
                      cursor: 'pointer',
                      '&:hover': { bgcolor: 'rgba(59, 130, 246, 0.03)' },
                    }}
                  >
                    <TableCell sx={{ py: 1.5 }}>
                      <Box>
                        <Tooltip title={etf.name}>
                          <Typography
                            sx={{
                              fontWeight: 600,
                              fontSize: '0.8rem',
                              color: '#1e293b',
                              maxWidth: 160,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {etf.name}
                          </Typography>
                        </Tooltip>
                        <Typography
                          sx={{
                            fontSize: '0.65rem',
                            color: '#94a3b8',
                            fontFamily: 'JetBrains Mono',
                          }}
                        >
                          {etf.code}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell align="right" sx={{ py: 1.5 }}>
                      <Typography
                        sx={{
                          fontSize: '0.85rem',
                          fontWeight: 700,
                          fontFamily: 'JetBrains Mono',
                          color: etf.change_pct > 0 ? '#22c55e' : etf.change_pct < 0 ? '#ef4444' : '#475569',
                        }}
                      >
                        {etf.price.toFixed(3)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right" sx={{ py: 1.5 }}>
                      {renderChangeValue(etf.change_pct)}
                    </TableCell>
                    <TableCell align="right" sx={{ py: 1.5 }}>
                      <Typography
                        sx={{
                          fontSize: '0.8rem',
                          fontFamily: 'JetBrains Mono',
                          color: '#475569',
                        }}
                      >
                        {formatAmount(etf.amount)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right" sx={{ py: 1.5 }}>
                      <Typography
                        sx={{
                          fontSize: '0.75rem',
                          fontFamily: 'JetBrains Mono',
                          color: '#64748b',
                        }}
                      >
                        {formatVolume(etf.volume)}
                      </Typography>
                    </TableCell>
                    <TableCell align="right" sx={{ py: 1.5 }}>
                      <Typography
                        sx={{
                          fontSize: '0.75rem',
                          fontFamily: 'JetBrains Mono',
                          color: '#64748b',
                        }}
                      >
                        {etf.turnover_rate.toFixed(2)}%
                      </Typography>
                    </TableCell>
                    <TableCell align="right" sx={{ py: 1.5 }}>
                      <Typography
                        sx={{
                          fontSize: '0.75rem',
                          fontFamily: 'JetBrains Mono',
                          color: amplitude > 3 ? '#f59e0b' : '#64748b',
                        }}
                      >
                        {amplitude.toFixed(2)}%
                      </Typography>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
}
