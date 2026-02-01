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
  CircularProgress,
  Chip,
  IconButton,
  Tooltip,
  FormControl,
  Select,
  MenuItem,
  Skeleton,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material/Select';
import RefreshIcon from '@mui/icons-material/Refresh';
import EmojiEventsIcon from '@mui/icons-material/EmojiEvents';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

import { fetchFundRanking,  } from '../../api';
import type{ FundRankingResponse, FundRankingItem } from '../../api';
const FUND_TYPES = [
  { key: '股票型', label: '股票型' },
  { key: '混合型', label: '混合型' },
  { key: '债券型', label: '债券型' },
  { key: '指数型', label: '指数型' },
  { key: 'QDII', label: 'QDII' },
  { key: 'FOF', label: 'FOF' },
];

const SORT_OPTIONS = [
  { key: '近1周', label: '近1周' },
  { key: '近1月', label: '近1月' },
  { key: '近3月', label: '近3月' },
  { key: '近6月', label: '近6月' },
  { key: '近1年', label: '近1年' },
  { key: '近3年', label: '近3年' },
];

interface Props {
  initialFundType?: string;
  onFundClick?: (code: string, name: string) => void;
}

export default function FundRankingTable({ initialFundType = '股票型', onFundClick }: Props) {
  const [data, setData] = useState<FundRankingResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [fundType, setFundType] = useState(initialFundType);
  const [sortBy, setSortBy] = useState('近1月');

  const loadData = async () => {
    setLoading(true);
    try {
      const result = await fetchFundRanking(fundType, sortBy, 50);
      setData(result);
    } catch (err: any) {
      console.error('Failed to load fund ranking:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [fundType, sortBy]);

  useEffect(() => {
    setFundType(initialFundType);
  }, [initialFundType]);

  const handleFundTypeChange = (e: SelectChangeEvent) => {
    setFundType(e.target.value);
  };

  const handleSortChange = (e: SelectChangeEvent) => {
    setSortBy(e.target.value);
  };

  const renderReturnValue = (value: number | undefined) => {
    if (value === undefined || value === null) return '-';
    const isPositive = value > 0;
    const color = isPositive ? '#22c55e' : value < 0 ? '#ef4444' : '#64748b';
    
    return (
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
    );
  };

  const getRankChip = (rank: number) => {
    if (rank === 1) {
      return (
        <Chip
          icon={<EmojiEventsIcon sx={{ fontSize: 14 }} />}
          label="1"
          size="small"
          sx={{
            bgcolor: '#fef3c7',
            color: '#d97706',
            fontWeight: 800,
            fontSize: '0.7rem',
            height: 24,
            '& .MuiChip-icon': { color: '#d97706' },
          }}
        />
      );
    } else if (rank === 2) {
      return (
        <Chip
          label="2"
          size="small"
          sx={{
            bgcolor: '#f1f5f9',
            color: '#64748b',
            fontWeight: 800,
            fontSize: '0.7rem',
            height: 24,
          }}
        />
      );
    } else if (rank === 3) {
      return (
        <Chip
          label="3"
          size="small"
          sx={{
            bgcolor: '#fef2f2',
            color: '#f97316',
            fontWeight: 800,
            fontSize: '0.7rem',
            height: 24,
          }}
        />
      );
    }
    return (
      <Typography
        sx={{
          fontWeight: 600,
          fontSize: '0.8rem',
          color: '#94a3b8',
          fontFamily: 'JetBrains Mono',
          pl: 1,
        }}
      >
        {rank}
      </Typography>
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
          <EmojiEventsIcon sx={{ color: '#f59e0b' }} />
          <Typography sx={{ fontWeight: 700, color: '#1e293b' }}>
            基金排行榜
          </Typography>
          {data && (
            <Chip
              label={`共 ${data.total.toLocaleString()} 只`}
              size="small"
              sx={{
                fontSize: '0.65rem',
                height: 20,
                bgcolor: '#f1f5f9',
                color: '#64748b',
              }}
            />
          )}
        </Box>

        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
          <FormControl size="small">
            <Select
              value={fundType}
              onChange={handleFundTypeChange}
              sx={{
                fontSize: '0.8rem',
                '& .MuiSelect-select': { py: 0.75, px: 1.5 },
                borderRadius: '8px',
              }}
            >
              {FUND_TYPES.map((t) => (
                <MenuItem key={t.key} value={t.key} sx={{ fontSize: '0.8rem' }}>
                  {t.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <FormControl size="small">
            <Select
              value={sortBy}
              onChange={handleSortChange}
              sx={{
                fontSize: '0.8rem',
                '& .MuiSelect-select': { py: 0.75, px: 1.5 },
                borderRadius: '8px',
              }}
            >
              {SORT_OPTIONS.map((s) => (
                <MenuItem key={s.key} value={s.key} sx={{ fontSize: '0.8rem' }}>
                  按{s.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

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
        <TableContainer sx={{ maxHeight: 600 }}>
          <Table stickyHeader size="small">
            <TableHead>
              <TableRow>
                <TableCell
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                    width: 60,
                  }}
                >
                  排名
                </TableCell>
                <TableCell
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  基金名称
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
                  净值
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: sortBy === '近1周' ? '#6366f1' : '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  近1周
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: sortBy === '近1月' ? '#6366f1' : '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  近1月
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: sortBy === '近3月' ? '#6366f1' : '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  近3月
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontWeight: 800,
                    fontSize: '0.7rem',
                    color: sortBy === '近1年' ? '#6366f1' : '#64748b',
                    bgcolor: '#f8fafc',
                  }}
                >
                  近1年
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
                  手续费
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {data?.funds.map((fund) => (
                <TableRow
                  key={fund.code}
                  hover
                  onClick={() => onFundClick?.(fund.code, fund.name)}
                  sx={{
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'rgba(99, 102, 241, 0.03)' },
                  }}
                >
                  <TableCell sx={{ py: 1.5 }}>{getRankChip(fund.rank)}</TableCell>
                  <TableCell sx={{ py: 1.5 }}>
                    <Box>
                      <Tooltip title={fund.name}>
                        <Typography
                          sx={{
                            fontWeight: 600,
                            fontSize: '0.8rem',
                            color: '#1e293b',
                            maxWidth: 180,
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                        >
                          {fund.name}
                        </Typography>
                      </Tooltip>
                      <Typography
                        sx={{
                          fontSize: '0.65rem',
                          color: '#94a3b8',
                          fontFamily: 'JetBrains Mono',
                        }}
                      >
                        {fund.code}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right" sx={{ py: 1.5 }}>
                    <Typography
                      sx={{
                        fontSize: '0.8rem',
                        fontFamily: 'JetBrains Mono',
                        color: '#475569',
                      }}
                    >
                      {fund.nav.toFixed(4)}
                    </Typography>
                  </TableCell>
                  <TableCell align="right" sx={{ py: 1.5 }}>
                    {renderReturnValue(fund.return_1w)}
                  </TableCell>
                  <TableCell align="right" sx={{ py: 1.5 }}>
                    {renderReturnValue(fund.return_1m)}
                  </TableCell>
                  <TableCell align="right" sx={{ py: 1.5 }}>
                    {renderReturnValue(fund.return_3m)}
                  </TableCell>
                  <TableCell align="right" sx={{ py: 1.5 }}>
                    {renderReturnValue(fund.return_1y)}
                  </TableCell>
                  <TableCell align="right" sx={{ py: 1.5 }}>
                    <Chip
                      label={fund.fee || '-'}
                      size="small"
                      sx={{
                        fontSize: '0.6rem',
                        height: 18,
                        bgcolor: '#f1f5f9',
                        color: '#64748b',
                      }}
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Paper>
  );
}
