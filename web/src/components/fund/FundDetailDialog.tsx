import { useState, useEffect } from 'react';
import {
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Grid,
  Tabs,
  Tab,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableRow,
} from '@mui/material';
import PieChartIcon from '@mui/icons-material/PieChart';
import PersonIcon from '@mui/icons-material/Person';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import { fetchFundFullDetail } from '../../api';
import type { FundFullDetail } from '../../api';

interface Props {
  open: boolean;
  onClose: () => void;
  fundCode: string;
  fundName: string;
}

export default function FundDetailDialog({ open, onClose, fundCode, fundName }: Props) {
  const [loading, setLoading] = useState(true);
  const [detail, setDetail] = useState<FundFullDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    if (open && fundCode) {
      loadDetail();
    }
  }, [open, fundCode]);

  const loadDetail = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchFundFullDetail(fundCode);
      setDetail(data);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : '加载基金详情失败';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const formatPercentage = (value: number | undefined | null, showSign: boolean = true) => {
    if (value === null || value === undefined) return '---';
    const formatted = value.toFixed(2);
    if (showSign && value > 0) return `+${formatted}%`;
    return `${formatted}%`;
  };

  const getColorByValue = (value: number | undefined | null) => {
    if (value === null || value === undefined) return '#94a3b8';
    return value > 0 ? '#22c55e' : value < 0 ? '#ef4444' : '#64748b';
  };

  // 净值走势图
  const renderNavChart = () => {
    if (!detail?.nav?.history || detail.nav.history.length < 2) {
      return (
        <Box sx={{ py: 6, textAlign: 'center', bgcolor: '#f8fafc', borderRadius: '12px' }}>
          <ShowChartIcon sx={{ fontSize: 40, color: '#e2e8f0', mb: 1 }} />
          <Typography sx={{ color: '#94a3b8', fontSize: '0.8rem' }}>暂无净值数据</Typography>
        </Box>
      );
    }

    const data = detail.nav.history;
    const values = data.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const padding = 20;
    const width = 600;
    const height = 180;

    const points = data.map((d, i) => {
      const x = (i / (data.length - 1)) * (width - 2 * padding) + padding;
      const y = (height - padding) - ((d.value - min) / range) * (height - 2 * padding);
      return `${x},${y}`;
    }).join(' ');

    const isUp = data[data.length - 1].value >= data[0].value;
    const lineColor = isUp ? '#22c55e' : '#ef4444';

    return (
      <Box sx={{ bgcolor: '#fcfcfc', p: 2, borderRadius: '12px', border: '1px solid #f1f5f9' }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 150 }}>
          {[0, 0.5, 1].map((v) => {
            const y = (height - padding) - v * (height - 2 * padding);
            const val = (min + v * range).toFixed(4);
            return (
              <g key={v}>
                <line x1={padding} y1={y} x2={width - padding} y2={y} stroke="#e2e8f0" strokeDasharray="4 4" />
                <text x={0} y={y + 4} fontSize="9" fill="#94a3b8" fontFamily="JetBrains Mono">{val}</text>
              </g>
            );
          })}
          <polyline
            fill="none"
            stroke={lineColor}
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={points}
          />
          <path
            d={`M${padding},${height - padding} L${points} L${width - padding},${height - padding} Z`}
            fill={`${lineColor}15`}
          />
        </svg>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
          <Typography sx={{ fontSize: '0.65rem', color: '#94a3b8' }}>
            {data[0].date}
          </Typography>
          <Typography sx={{ fontSize: '0.65rem', color: '#94a3b8' }}>
            {data[data.length - 1].date}
          </Typography>
        </Box>
      </Box>
    );
  };

  // 基本信息Tab
  const renderBasicInfo = () => (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 净值信息 */}
      <Box>
        <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
          <ShowChartIcon sx={{ fontSize: 16 }} /> 净值信息
        </Typography>
        <Grid container spacing={2}>
          <Grid size={4}>
            <Box sx={{ p: 2, bgcolor: '#f8fafc', borderRadius: '12px', textAlign: 'center' }}>
              <Typography sx={{ fontSize: '0.65rem', color: '#94a3b8', mb: 0.5 }}>单位净值</Typography>
              <Typography sx={{ fontSize: '1.3rem', fontWeight: 800, color: '#1e293b', fontFamily: 'JetBrains Mono' }}>
                {detail?.nav?.current?.toFixed(4) || '---'}
              </Typography>
            </Box>
          </Grid>
          <Grid size={4}>
            <Box sx={{ p: 2, bgcolor: '#f8fafc', borderRadius: '12px', textAlign: 'center' }}>
              <Typography sx={{ fontSize: '0.65rem', color: '#94a3b8', mb: 0.5 }}>累计净值</Typography>
              <Typography sx={{ fontSize: '1.3rem', fontWeight: 800, color: '#6366f1', fontFamily: 'JetBrains Mono' }}>
                {detail?.nav?.accumulated?.toFixed(4) || '---'}
              </Typography>
            </Box>
          </Grid>
          <Grid size={4}>
            <Box sx={{ p: 2, bgcolor: '#f8fafc', borderRadius: '12px', textAlign: 'center' }}>
              <Typography sx={{ fontSize: '0.65rem', color: '#94a3b8', mb: 0.5 }}>费率</Typography>
              <Typography sx={{ fontSize: '1.3rem', fontWeight: 800, color: '#f59e0b', fontFamily: 'JetBrains Mono' }}>
                {detail?.performance?.fee || '---'}
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </Box>

      {/* 净值走势 */}
      <Box>
        <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', mb: 1.5 }}>
          📈 近期净值走势
        </Typography>
        {renderNavChart()}
      </Box>

      {/* 基金信息 */}
      <Box>
        <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
          <InfoIcon sx={{ fontSize: 16 }} /> 基金信息
        </Typography>
        <Box sx={{ p: 2, borderRadius: '12px', border: '1px solid #f1f5f9' }}>
          <Grid container spacing={2}>
            {[
              { label: '基金公司', value: detail?.basic_info?.company },
              { label: '基金类型', value: detail?.basic_info?.fund_type },
              { label: '成立日期', value: detail?.basic_info?.inception_date },
              { label: '基金规模', value: detail?.basic_info?.size },
            ].map((item, idx) => (
              <Grid size={6} key={idx}>
                <Typography sx={{ fontSize: '0.65rem', color: '#94a3b8', mb: 0.5 }}>{item.label}</Typography>
                <Typography sx={{ fontSize: '0.8rem', fontWeight: 600, color: '#334155' }}>
                  {item.value || '---'}
                </Typography>
              </Grid>
            ))}
          </Grid>
        </Box>
      </Box>
    </Box>
  );

  // 业绩表现Tab
  const renderPerformance = () => (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 收益率网格 */}
      <Box>
        <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', mb: 1.5 }}>
          📊 阶段收益率
        </Typography>
        <Grid container spacing={1.5}>
          {[
            { label: '近1周', value: detail?.performance?.return_1w },
            { label: '近1月', value: detail?.performance?.return_1m },
            { label: '近3月', value: detail?.performance?.return_3m },
            { label: '近6月', value: detail?.performance?.return_6m },
            { label: '近1年', value: detail?.performance?.return_1y },
            { label: '近2年', value: detail?.performance?.return_2y },
            { label: '近3年', value: detail?.performance?.return_3y },
            { label: '今年以来', value: detail?.performance?.return_ytd },
            { label: '成立以来', value: detail?.performance?.return_since_inception },
          ].map((item, idx) => (
            <Grid size={4} key={idx}>
              <Box
                sx={{
                  p: 1.5,
                  borderRadius: '10px',
                  border: '1px solid #f1f5f9',
                  textAlign: 'center',
                  bgcolor: item.value && item.value > 0 ? '#f0fdf4' : item.value && item.value < 0 ? '#fef2f2' : '#f8fafc',
                }}
              >
                <Typography sx={{ fontSize: '0.6rem', color: '#64748b', mb: 0.5 }}>{item.label}</Typography>
                <Typography
                  sx={{
                    fontSize: '0.9rem',
                    fontWeight: 800,
                    color: getColorByValue(item.value),
                    fontFamily: 'JetBrains Mono',
                  }}
                >
                  {formatPercentage(item.value)}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Box>

      {/* 风险指标 */}
      <Box>
        <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
          <WarningIcon sx={{ fontSize: 16, color: '#f59e0b' }} /> 风险指标
        </Typography>
        <TableContainer sx={{ borderRadius: '12px', border: '1px solid #f1f5f9' }}>
          <Table size="small">
            <TableBody>
              {[
                { label: '最大回撤', value: detail?.risk_metrics?.max_drawdown, suffix: '%', color: '#ef4444' },
                { label: '夏普比率', value: detail?.risk_metrics?.sharpe_ratio, suffix: '', color: '#6366f1' },
                { label: '年化波动率', value: detail?.risk_metrics?.volatility, suffix: '%', color: '#f59e0b' },
                { label: '年化收益率', value: detail?.risk_metrics?.annualized_return, suffix: '%' },
                { label: '卡尔玛比率', value: detail?.risk_metrics?.calmar_ratio, suffix: '', color: '#22c55e' },
                { label: '索提诺比率', value: detail?.risk_metrics?.sortino_ratio, suffix: '', color: '#3b82f6' },
              ].map((item, idx) => (
                <TableRow key={idx} sx={{ '&:last-child td': { border: 0 } }}>
                  <TableCell sx={{ color: '#64748b', fontWeight: 600, fontSize: '0.8rem' }}>
                    {item.label}
                  </TableCell>
                  <TableCell align="right" sx={{ fontWeight: 800, fontFamily: 'JetBrains Mono', color: item.color || getColorByValue(item.value) }}>
                    {item.value !== null && item.value !== undefined && typeof item.value === 'number' ? `${item.value.toFixed(2)}${item.suffix}` : '---'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Box>
  );

  // 持仓分析Tab
  const renderHoldings = () => (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 重仓股票 */}
      <Box>
        <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
          <AccountBalanceIcon sx={{ fontSize: 16 }} /> 重仓股票 Top 10
        </Typography>
        {detail?.holdings && detail.holdings.length > 0 ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {detail.holdings.map((stock, idx) => (
              <Box
                key={idx}
                sx={{
                  p: 1.5,
                  borderRadius: '10px',
                  border: '1px solid #f1f5f9',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  '&:hover': { bgcolor: '#f8fafc' },
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                  <Chip
                    label={idx + 1}
                    size="small"
                    sx={{
                      width: 24,
                      height: 24,
                      fontSize: '0.65rem',
                      fontWeight: 800,
                      bgcolor: idx < 3 ? '#fef3c7' : '#f1f5f9',
                      color: idx < 3 ? '#d97706' : '#64748b',
                    }}
                  />
                  <Box>
                    <Typography sx={{ fontSize: '0.8rem', fontWeight: 700, color: '#1e293b' }}>
                      {stock.name}
                    </Typography>
                    <Typography sx={{ fontSize: '0.65rem', color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>
                      {stock.code}
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ textAlign: 'right', display: 'flex', alignItems: 'center', gap: 1.5 }}>
                  <Typography sx={{ fontSize: '0.85rem', fontWeight: 800, color: '#6366f1', fontFamily: 'JetBrains Mono' }}>
                    {stock.weight.toFixed(2)}%
                  </Typography>
                  <Box sx={{ width: 50, height: 6, bgcolor: '#f1f5f9', borderRadius: 3, overflow: 'hidden' }}>
                    <Box sx={{ width: `${Math.min(stock.weight * 10, 100)}%`, height: '100%', bgcolor: '#6366f1', borderRadius: 3 }} />
                  </Box>
                </Box>
              </Box>
            ))}
          </Box>
        ) : (
          <Box sx={{ py: 6, textAlign: 'center', bgcolor: '#f8fafc', borderRadius: '12px' }}>
            <PieChartIcon sx={{ fontSize: 40, color: '#e2e8f0', mb: 1 }} />
            <Typography sx={{ color: '#94a3b8', fontSize: '0.8rem' }}>暂无持仓数据</Typography>
          </Box>
        )}
      </Box>

      {/* 行业配置 */}
      <Box>
        <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
          <PieChartIcon sx={{ fontSize: 16 }} /> 行业配置
        </Typography>
        {detail?.industry_allocation && detail.industry_allocation.length > 0 ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {detail.industry_allocation.map((industry, idx) => (
              <Box
                key={idx}
                sx={{
                  p: 1.5,
                  borderRadius: '10px',
                  border: '1px solid #f1f5f9',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                }}
              >
                <Typography sx={{ fontSize: '0.8rem', fontWeight: 600, color: '#334155' }}>
                  {industry.industry}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                  <Box sx={{ width: 80, height: 6, bgcolor: '#f1f5f9', borderRadius: 3, overflow: 'hidden' }}>
                    <Box
                      sx={{
                        width: `${Math.min(industry.weight, 100)}%`,
                        height: '100%',
                        bgcolor: `hsl(${240 - idx * 30}, 70%, 60%)`,
                        borderRadius: 3,
                      }}
                    />
                  </Box>
                  <Typography sx={{ fontSize: '0.8rem', fontWeight: 700, color: '#6366f1', fontFamily: 'JetBrains Mono', minWidth: 50, textAlign: 'right' }}>
                    {industry.weight.toFixed(2)}%
                  </Typography>
                </Box>
              </Box>
            ))}
          </Box>
        ) : (
          <Box sx={{ py: 6, textAlign: 'center', bgcolor: '#f8fafc', borderRadius: '12px' }}>
            <PieChartIcon sx={{ fontSize: 40, color: '#e2e8f0', mb: 1 }} />
            <Typography sx={{ color: '#94a3b8', fontSize: '0.8rem' }}>暂无行业配置数据</Typography>
          </Box>
        )}
      </Box>
    </Box>
  );

  // 基金经理Tab
  const renderManagers = () => (
    <Box>
      <Typography sx={{ fontSize: '0.75rem', fontWeight: 700, color: '#64748b', mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
        <PersonIcon sx={{ fontSize: 16 }} /> 基金经理
      </Typography>
      {detail?.managers && detail.managers.length > 0 ? (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {detail.managers.map((manager, idx) => (
            <Box
              key={idx}
              sx={{
                p: 2,
                borderRadius: '12px',
                border: '1px solid #f1f5f9',
                bgcolor: idx === 0 ? '#fafbff' : '#fff',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1.5 }}>
                <Box
                  sx={{
                    width: 48,
                    height: 48,
                    borderRadius: '50%',
                    bgcolor: manager.gender === '女' ? '#ec4899' : '#6366f1',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                  }}
                >
                  <PersonIcon sx={{ color: '#fff', fontSize: 24 }} />
                </Box>
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                    <Typography sx={{ fontWeight: 800, color: '#1e293b', fontSize: '0.95rem' }}>
                      {manager.name}
                    </Typography>
                    {manager.gender && (
                      <Box
                        sx={{
                          px: 0.75,
                          py: 0.25,
                          borderRadius: '4px',
                          bgcolor: manager.gender === '女' ? '#fdf2f8' : '#eff6ff',
                          color: manager.gender === '女' ? '#db2777' : '#3b82f6',
                          fontSize: '0.65rem',
                          fontWeight: 600,
                        }}
                      >
                        {manager.gender}
                      </Box>
                    )}
                    {manager.education && (
                      <Box
                        sx={{
                          px: 0.75,
                          py: 0.25,
                          borderRadius: '4px',
                          bgcolor: '#f0fdf4',
                          color: '#16a34a',
                          fontSize: '0.65rem',
                          fontWeight: 600,
                        }}
                      >
                        {manager.education}
                      </Box>
                    )}
                    {manager.birth_year && (
                      <Box
                        sx={{
                          px: 0.75,
                          py: 0.25,
                          borderRadius: '4px',
                          bgcolor: '#fefce8',
                          color: '#ca8a04',
                          fontSize: '0.65rem',
                          fontWeight: 600,
                        }}
                      >
                        {manager.birth_year}年生
                      </Box>
                    )}
                  </Box>
                  <Typography sx={{ fontSize: '0.7rem', color: '#94a3b8', mt: 0.5 }}>
                    任职时间: {manager.start_date || '-'} ~ {manager.end_date || '至今'}
                  </Typography>
                </Box>
              </Box>
              
              <Grid container spacing={1.5} sx={{ mb: manager.resume ? 1.5 : 0 }}>
                <Grid size={6}>
                  <Box sx={{ p: 1.5, bgcolor: '#f8fafc', borderRadius: '8px', textAlign: 'center' }}>
                    <Typography sx={{ fontSize: '0.6rem', color: '#94a3b8', mb: 0.5 }}>任职天数</Typography>
                    <Typography sx={{ fontSize: '1rem', fontWeight: 800, color: '#334155', fontFamily: 'JetBrains Mono' }}>
                      {manager.tenure_days > 0 ? `${manager.tenure_days} 天` : '-'}
                    </Typography>
                    {manager.tenure_days > 0 && (
                      <Typography sx={{ fontSize: '0.6rem', color: '#94a3b8' }}>
                        约 {(manager.tenure_days / 365).toFixed(1)} 年
                      </Typography>
                    )}
                  </Box>
                </Grid>
                <Grid size={6}>
                  <Box sx={{ p: 1.5, bgcolor: manager.tenure_return >= 0 ? '#f0fdf4' : '#fef2f2', borderRadius: '8px', textAlign: 'center' }}>
                    <Typography sx={{ fontSize: '0.6rem', color: '#94a3b8', mb: 0.5 }}>任期回报</Typography>
                    <Typography
                      sx={{
                        fontSize: '1rem',
                        fontWeight: 800,
                        color: getColorByValue(manager.tenure_return),
                        fontFamily: 'JetBrains Mono',
                      }}
                    >
                      {manager.tenure_return !== 0 ? formatPercentage(manager.tenure_return) : '-'}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
              
              {/* Manager Resume */}
              {manager.resume && (
                <Box
                  sx={{
                    p: 1.5,
                    bgcolor: '#f8fafc',
                    borderRadius: '8px',
                    borderLeft: '3px solid #6366f1',
                  }}
                >
                  <Typography sx={{ fontSize: '0.6rem', color: '#94a3b8', mb: 0.5, fontWeight: 600 }}>
                    经理简介
                  </Typography>
                  <Typography
                    sx={{
                      fontSize: '0.75rem',
                      color: '#475569',
                      lineHeight: 1.6,
                      display: '-webkit-box',
                      WebkitLineClamp: 4,
                      WebkitBoxOrient: 'vertical',
                      overflow: 'hidden',
                    }}
                  >
                    {manager.resume}
                  </Typography>
                </Box>
              )}
            </Box>
          ))}
        </Box>
      ) : (
        <Box sx={{ py: 6, textAlign: 'center', bgcolor: '#f8fafc', borderRadius: '12px' }}>
          <PersonIcon sx={{ fontSize: 40, color: '#e2e8f0', mb: 1 }} />
          <Typography sx={{ color: '#94a3b8', fontSize: '0.8rem' }}>暂无基金经理数据</Typography>
        </Box>
      )}
    </Box>
  );

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      scroll="paper"
      PaperProps={{
        sx: {
          borderRadius: '20px',
          bgcolor: '#ffffff',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          maxHeight: '90vh',
        },
      }}
    >
      {/* Header */}
      <DialogTitle sx={{ p: 0 }}>
        <Box
          sx={{
            bgcolor: '#fcfcfc',
            p: 3,
            borderBottom: '1px solid #f1f5f9',
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <Box sx={{ flex: 1 }}>
              <Typography
                sx={{
                  fontWeight: 900,
                  color: '#0f172a',
                  fontSize: '1.25rem',
                  letterSpacing: '-0.02em',
                  mb: 1,
                  lineHeight: 1.2,
                }}
              >
                {fundName}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center' }}>
                <Box sx={{ px: 1, py: 0.25, bgcolor: '#6366f1', borderRadius: '4px' }}>
                  <Typography sx={{ color: '#fff', fontWeight: 800, fontFamily: 'JetBrains Mono', fontSize: '0.75rem' }}>
                    {fundCode}
                  </Typography>
                </Box>
                {detail?.basic_info?.fund_type && (
                  <Chip
                    label={detail.basic_info.fund_type}
                    size="small"
                    sx={{ bgcolor: '#f1f5f9', color: '#64748b', fontWeight: 600, fontSize: '0.7rem' }}
                  />
                )}
              </Box>
            </Box>
            {detail?.nav && (
              <Box sx={{ textAlign: 'right' }}>
                <Typography sx={{ color: '#6366f1', fontSize: '0.65rem', fontWeight: 900, mb: 0.5, letterSpacing: '0.1em' }}>
                  最新净值
                </Typography>
                <Typography
                  sx={{
                    color: '#0f172a',
                    fontSize: '1.5rem',
                    fontWeight: 900,
                    fontFamily: 'JetBrains Mono',
                    lineHeight: 1,
                  }}
                >
                  {detail.nav.current?.toFixed(4) || '---'}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ p: 0, bgcolor: '#ffffff' }}>
        {loading ? (
          <Box sx={{ py: 12, textAlign: 'center' }}>
            <CircularProgress size={32} thickness={5} sx={{ color: '#6366f1', mb: 2 }} />
            <Typography sx={{ color: '#64748b', fontWeight: 600, fontSize: '0.85rem' }}>
              加载基金详情...
            </Typography>
          </Box>
        ) : error ? (
          <Box sx={{ py: 12, textAlign: 'center' }}>
            <WarningIcon sx={{ fontSize: 40, color: '#f59e0b', mb: 1 }} />
            <Typography sx={{ color: '#ef4444', fontWeight: 600 }}>{error}</Typography>
            <Button onClick={loadDetail} sx={{ mt: 2, color: '#6366f1' }}>
              重试
            </Button>
          </Box>
        ) : (
          <Box sx={{ p: 3 }}>
            <Tabs
              value={activeTab}
              onChange={(_, v) => setActiveTab(v)}
              sx={{
                minHeight: '40px',
                mb: 3,
                borderBottom: '1px solid #f1f5f9',
                '& .MuiTab-root': {
                  py: 1,
                  minHeight: '40px',
                  textTransform: 'none',
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  color: '#94a3b8',
                },
                '& .Mui-selected': { color: '#6366f1 !important' },
                '& .MuiTabs-indicator': { bgcolor: '#6366f1', height: 3, borderRadius: '3px 3px 0 0' },
              }}
            >
              <Tab label="基本信息" />
              <Tab label="业绩表现" />
              <Tab label="持仓分析" />
              <Tab label="基金经理" />
            </Tabs>

            {activeTab === 0 && renderBasicInfo()}
            {activeTab === 1 && renderPerformance()}
            {activeTab === 2 && renderHoldings()}
            {activeTab === 3 && renderManagers()}
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 3, bgcolor: '#fcfcfc', borderTop: '1px solid #f1f5f9' }}>
        <Button
          fullWidth
          onClick={onClose}
          variant="contained"
          sx={{
            bgcolor: '#0f172a',
            color: '#ffffff',
            py: 1.5,
            borderRadius: '12px',
            textTransform: 'none',
            fontWeight: 800,
            fontSize: '0.9rem',
            boxShadow: 'none',
            '&:hover': { bgcolor: '#1e293b', boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)' },
          }}
        >
          关闭
        </Button>
      </DialogActions>
    </Dialog>
  );
}
