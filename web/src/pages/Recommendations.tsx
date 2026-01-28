import { useEffect, useState, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
    Box,
    Typography,
    CircularProgress,
    Chip,
    IconButton,
    Paper,
    Button,
    ToggleButton,
    ToggleButtonGroup,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Tabs,
    Tab,
    Alert,
    Tooltip,
    Collapse,
    Snackbar,
    Badge,
    Popover,
    Fade,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import PieChartIcon from '@mui/icons-material/PieChart';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import InsightsIcon from '@mui/icons-material/Insights';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import TuneIcon from '@mui/icons-material/Tune';
import CloseIcon from '@mui/icons-material/Close';
import LightbulbIcon from '@mui/icons-material/Lightbulb';
import ScienceIcon from '@mui/icons-material/Science';
import {
    generateRecommendations,
    fetchLatestRecommendations,
    getUserPreferences,
    pollRecommendationTask,
    generateRecommendationsV2,
    getFactorStatusV2,
} from '../api';

import type {
    RecommendationResult,
    RecommendationStock,
    RecommendationFund,
    TaskStatusResponse,
    RecommendationResultV2,
    RecommendationStockV2,
    RecommendationFundV2,
    FactorStatus,
} from '../api';

import PreferencesModal from '../components/PreferencesModal';

// --- Utility Components ---

const NumberMono = ({ children, className = "", style = {} }: { children: React.ReactNode, className?: string, style?: React.CSSProperties }) => (
    <span className={`font-mono tracking-tight ${className}`} style={{ ...style, fontVariantNumeric: 'tabular-nums' }}>
        {children}
    </span>
);

const ColorVal = ({ val, suffix = "", bold = true }: { val: number | null | undefined, suffix?: string, bold?: boolean }) => {
    if (val === null || val === undefined) return <span className="text-slate-400">-</span>;
    const colorClass = val > 0 ? "text-red-600" : val < 0 ? "text-green-600" : "text-slate-500";
    return (
        <NumberMono className={`${bold ? 'font-semibold' : ''} ${colorClass}`}>
            {val > 0 ? '+' : ''}{typeof val === 'number' ? val.toFixed(2) : val}{suffix}
        </NumberMono>
    );
};

const ScoreBar = ({ score, maxScore = 100 }: { score: number, maxScore?: number }) => {
    const percentage = Math.min((score / maxScore) * 100, 100);
    const getColor = () => {
        if (percentage >= 70) return 'bg-green-500';
        if (percentage >= 50) return 'bg-blue-500';
        if (percentage >= 30) return 'bg-yellow-500';
        return 'bg-red-500';
    };
    return (
        <Box className="flex items-center gap-2 w-full">
            <Box className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                <Box className={`h-full ${getColor()} rounded-full transition-all`} style={{ width: `${percentage}%` }} />
            </Box>
            <NumberMono className="text-xs font-semibold text-slate-600 w-8 text-right">{score.toFixed(0)}</NumberMono>
        </Box>
    );
};

const ConfidenceChip = ({ confidence }: { confidence?: string }) => {
    const { t } = useTranslation();
    if (!confidence) return null;
    const level = confidence.toLowerCase();
    const config: Record<string, { bg: string, text: string, label: string }> = {
        '高': { bg: 'bg-green-50', text: 'text-green-700', label: t('recommendations.confidence_levels.high') },
        'high': { bg: 'bg-green-50', text: 'text-green-700', label: t('recommendations.confidence_levels.high') },
        '中': { bg: 'bg-blue-50', text: 'text-blue-700', label: t('recommendations.confidence_levels.medium') },
        'medium': { bg: 'bg-blue-50', text: 'text-blue-700', label: t('recommendations.confidence_levels.medium') },
        '低': { bg: 'bg-orange-50', text: 'text-orange-700', label: t('recommendations.confidence_levels.low') },
        'low': { bg: 'bg-orange-50', text: 'text-orange-700', label: t('recommendations.confidence_levels.low') },
    };
    const c = config[level] || config['medium'];
    return <Chip label={c.label} size="small" className={`h-5 text-[10px] font-bold ${c.bg} ${c.text}`} />;
};

const formatMarketCap = (cap: number | string | null | undefined): string => {
    if (cap === null || cap === undefined || cap === '') return '-';
    const num = typeof cap === 'number' ? cap : parseFloat(String(cap));
    if (isNaN(num)) return '-';
    if (num >= 1e12) return `${(num / 1e12).toFixed(1)}万亿`;
    if (num >= 1e8) return `${(num / 1e8).toFixed(0)}亿`;
    return `${(num / 1e4).toFixed(0)}万`;
};

const formatAmount = (amount: number | string | null | undefined): string => {
    if (amount === null || amount === undefined || amount === '') return '-';
    const num = typeof amount === 'number' ? amount : parseFloat(String(amount));
    if (isNaN(num)) return '-';
    if (Math.abs(num) >= 1e8) return `${(num / 1e8).toFixed(2)}亿`;
    if (Math.abs(num) >= 1e4) return `${(num / 1e4).toFixed(0)}万`;
    return num.toFixed(0);
};

// --- Tab Preview Component ---

interface TabPreviewContentProps {
    type: 'stocks' | 'funds';
    stocks: RecommendationStock[];
    funds: RecommendationFund[];
    isShortTerm: boolean;
}

const TabPreviewContent = ({ type, stocks, funds, isShortTerm }: TabPreviewContentProps) => {
    const { t } = useTranslation();
    const items = type === 'stocks' ? stocks : funds;
    const avgScore = items.length > 0
        ? items.reduce((sum, item) => sum + ((item as any).recommendation_score || (item as any).score || 0), 0) / items.length
        : 0;

    const themeColor = type === 'stocks' ? {
        gradient: 'from-blue-500 to-cyan-500',
        bg: 'bg-blue-50',
        text: 'text-blue-700',
        border: 'border-blue-100',
        icon: <ShowChartIcon className="text-white text-base" />
    } : {
        gradient: 'from-purple-500 to-pink-500',
        bg: 'bg-purple-50',
        text: 'text-purple-700',
        border: 'border-purple-100',
        icon: <PieChartIcon className="text-white text-base" />
    };

    if (!items || items.length === 0) {
        return (
            <Box className="p-5 min-w-[280px]">
                <Box className="flex flex-col items-center gap-3 py-4">
                    <Box className={`w-12 h-12 rounded-xl bg-gradient-to-br ${themeColor.gradient} flex items-center justify-center opacity-50`}>
                        {themeColor.icon}
                    </Box>
                    <Typography variant="body2" className="text-slate-400">
                        {t('recommendations.no_data')}
                    </Typography>
                </Box>
            </Box>
        );
    }

    return (
        <Box className="min-w-[300px] max-w-[360px] overflow-hidden">
            {/* Header with gradient */}
            <Box className={`bg-gradient-to-r ${themeColor.gradient} px-4 py-3`}>
                <Box className="flex items-center justify-between">
                    <Box className="flex items-center gap-2">
                        <Box className="w-8 h-8 rounded-lg bg-white/20 backdrop-blur flex items-center justify-center">
                            {themeColor.icon}
                        </Box>
                        <Box>
                            <Typography variant="subtitle2" className="font-bold text-white">
                                {type === 'stocks' ? t('recommendations.tabs.stocks') : t('recommendations.tabs.funds')}
                            </Typography>
                            <Typography variant="caption" className="text-white/70 text-[10px]">
                                {isShortTerm ? t('recommendations.short_term.subtitle') : t('recommendations.long_term.subtitle')}
                            </Typography>
                        </Box>
                    </Box>
                    <Box className="bg-white/20 backdrop-blur rounded-full px-2.5 py-1">
                        <Typography variant="caption" className="text-white font-bold text-xs">
                            {items.length} {t('recommendations.preview.count')}
                        </Typography>
                    </Box>
                </Box>
            </Box>

            {/* Content */}
            <Box className="p-4">
                {/* Top 3 Items */}
                <Box className="space-y-2">
                    {items.slice(0, 3).map((item, i) => {
                        const score = (item as any).recommendation_score || (item as any).score || 0;
                        const changePct = (item as any).change_pct;
                        const return1w = (item as any).return_1w;
                        const displayChange = type === 'stocks' ? changePct : return1w;

                        return (
                            <Box
                                key={i}
                                className={`flex items-center gap-3 p-2.5 rounded-lg border ${themeColor.border} ${themeColor.bg} transition-all`}
                            >
                                {/* Rank Badge */}
                                <Box className={`w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold shadow-sm ${
                                    i === 0
                                        ? 'bg-gradient-to-br from-amber-400 to-orange-500 text-white'
                                        : i === 1
                                            ? 'bg-gradient-to-br from-slate-300 to-slate-400 text-white'
                                            : 'bg-slate-200 text-slate-600'
                                }`}>
                                    {i + 1}
                                </Box>

                                {/* Name & Code */}
                                <Box className="flex-1 min-w-0">
                                    <Typography variant="body2" className="font-semibold text-slate-800 truncate text-sm">
                                        {item.name}
                                    </Typography>
                                    <Typography variant="caption" className="text-slate-400 text-[10px]">
                                        {item.code}
                                    </Typography>
                                </Box>

                                {/* Score */}
                                <Box className="flex flex-col items-end">
                                    <NumberMono className={`text-sm font-bold ${themeColor.text}`}>
                                        {score.toFixed(0)}
                                    </NumberMono>
                                    {displayChange !== undefined && displayChange !== null && typeof displayChange === 'number' && (
                                        <NumberMono className={`text-[10px] font-medium ${
                                            displayChange > 0 ? 'text-red-500' : displayChange < 0 ? 'text-green-500' : 'text-slate-400'
                                        }`}>
                                            {displayChange > 0 ? '+' : ''}{displayChange.toFixed(2)}
                                        </NumberMono>
                                    )}
                                </Box>
                            </Box>
                        );
                    })}
                </Box>

                {/* More indicator */}
                {items.length > 3 && (
                    <Box className="flex items-center justify-center gap-1 mt-3">
                        <Box className="flex gap-0.5">
                            {[...Array(Math.min(items.length - 3, 3))].map((_, i) => (
                                <Box key={i} className="w-1.5 h-1.5 rounded-full bg-slate-300" />
                            ))}
                        </Box>
                        <Typography variant="caption" className="text-slate-400 ml-1">
                            {t('recommendations.preview.more')} {items.length - 3} {t('recommendations.preview.count')}
                        </Typography>
                    </Box>
                )}

                {/* Stats Footer */}
                <Box className={`mt-4 pt-3 border-t ${themeColor.border} flex items-center justify-between`}>
                    <Box className="flex items-center gap-4">
                        <Box className="flex items-center gap-1.5">
                            <Box className="w-2 h-2 rounded-full bg-green-500" />
                            <Typography variant="caption" className="text-slate-500">
                                {t('recommendations.preview.avg_score')} <span className={`font-bold ${themeColor.text}`}>{avgScore.toFixed(0)}</span>
                            </Typography>
                        </Box>
                    </Box>
                    <Box className={`flex items-center gap-1 ${themeColor.text} opacity-80`}>
                        <Typography variant="caption" className="font-medium">
                            {t('recommendations.preview.view_all')}
                        </Typography>
                        <Box className="text-xs">→</Box>
                    </Box>
                </Box>
            </Box>
        </Box>
    );
};

// --- Stock Detail Modal Component ---

interface StockDetailModalProps {
    open: boolean;
    onClose: () => void;
    stock: RecommendationStock | null;
    isShortTerm: boolean;
}

const StockDetailModal = ({ open, onClose, stock, isShortTerm }: StockDetailModalProps) => {
    const { t } = useTranslation();
    if (!stock) return null;

    const price = typeof stock.current_price === 'number' ? stock.current_price :
                  typeof stock.price === 'number' ? stock.price : null;

    const safeFixed = (val: number | string | undefined | null, digits: number = 2): string => {
        if (val === undefined || val === null) return '-';
        const num = typeof val === 'number' ? val : parseFloat(String(val));
        if (isNaN(num)) return '-';
        return num.toFixed(digits);
    };

    const changePct = Number(stock.change_pct) || 0;

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="sm"
            fullWidth
            PaperProps={{ sx: { borderRadius: '12px', maxHeight: '85vh' } }}
        >
            <DialogTitle sx={{ p: 0 }}>
                <Box sx={{ px: 2.5, py: 2, borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                                <Typography sx={{ fontWeight: 700, color: '#1e293b', fontSize: '1.1rem' }}>
                                    {stock.name}
                                </Typography>
                                <Typography sx={{ color: '#64748b', fontFamily: 'monospace', fontSize: '0.8rem' }}>
                                    {stock.code}
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 0.5 }}>
                                <Typography sx={{
                                    fontWeight: 700, fontFamily: 'monospace', fontSize: '1rem',
                                    color: changePct > 0 ? '#dc2626' : changePct < 0 ? '#16a34a' : '#64748b'
                                }}>
                                    ¥{price !== null ? price.toFixed(2) : '-'}
                                </Typography>
                                <Typography sx={{
                                    fontWeight: 600, fontFamily: 'monospace', fontSize: '0.85rem',
                                    color: changePct > 0 ? '#dc2626' : changePct < 0 ? '#16a34a' : '#64748b'
                                }}>
                                    {changePct > 0 ? '+' : ''}{safeFixed(stock.change_pct)}%
                                </Typography>
                                <ConfidenceChip confidence={stock.confidence} />
                            </Box>
                        </Box>
                    </Box>
                    <IconButton onClick={onClose} size="small" sx={{ color: '#94a3b8' }}>
                        <CloseIcon fontSize="small" />
                    </IconButton>
                </Box>
            </DialogTitle>

            <DialogContent sx={{ p: 0 }}>
                <Box sx={{ px: 2.5, py: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box>
                        <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.75, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <LightbulbIcon sx={{ fontSize: 14 }} /> {t('recommendations.detail.reason')}
                        </Typography>
                        <Typography sx={{ color: '#334155', fontSize: '0.85rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                            {stock.investment_logic || t('recommendations.detail.no_logic')}
                        </Typography>
                    </Box>

                    {stock.why_now && (
                        <Box>
                            <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.75, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <AccessTimeIcon sx={{ fontSize: 14 }} /> {t('recommendations.detail.why_now')}
                            </Typography>
                            <Typography sx={{ color: '#334155', fontSize: '0.85rem', lineHeight: 1.6 }}>
                                {stock.why_now}
                            </Typography>
                        </Box>
                    )}

                    <Box sx={{ bgcolor: '#f8fafc', borderRadius: '8px', p: 1.5 }}>
                        <Typography sx={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', mb: 1 }}>{t('recommendations.detail.key_data')}</Typography>
                        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1.5 }}>
                            <Box>
                                <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.pe')}</Typography>
                                <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                    {safeFixed(stock.pe, 1)}
                                </Typography>
                            </Box>
                            {stock.pb && (
                                <Box>
                                    <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.pb')}</Typography>
                                    <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                        {safeFixed(stock.pb, 2)}
                                    </Typography>
                                </Box>
                            )}
                            <Box>
                                <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.market_cap')}</Typography>
                                <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                    {formatMarketCap(stock.market_cap)}
                                </Typography>
                            </Box>
                            {isShortTerm && (
                                <>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.main_inflow')}</Typography>
                                        <Typography sx={{
                                            fontWeight: 600, fontFamily: 'monospace', fontSize: '0.85rem',
                                            color: (Number(stock.main_net_inflow) || 0) > 0 ? '#dc2626' : '#16a34a'
                                        }}>
                                            {formatAmount(stock.main_net_inflow)}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.volume_ratio')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {safeFixed(stock.volume_ratio, 2)}
                                        </Typography>
                                    </Box>
                                </>
                            )}
                        </Box>
                    </Box>

                    {isShortTerm && (stock.target_price || stock.stop_loss) && (
                        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1.5 }}>
                            {stock.target_price && (
                                <Box sx={{ bgcolor: '#f8fafc', borderRadius: '8px', p: 1.5 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.7rem', mb: 0.5 }}>{t('recommendations.detail.target_price')}</Typography>
                                    <Typography sx={{ fontWeight: 700, fontFamily: 'monospace', color: '#16a34a', fontSize: '1rem' }}>
                                        ¥{safeFixed(stock.target_price, 2)}
                                    </Typography>
                                    {price && typeof stock.target_price === 'number' && (
                                        <Typography sx={{ color: '#64748b', fontSize: '0.7rem' }}>
                                            +{((stock.target_price / price - 1) * 100).toFixed(1)}%
                                        </Typography>
                                    )}
                                </Box>
                            )}
                            {stock.stop_loss && (
                                <Box sx={{ bgcolor: '#f8fafc', borderRadius: '8px', p: 1.5 }}>
                                    <Typography sx={{ color: '#64748b', fontSize: '0.7rem', mb: 0.5 }}>{t('recommendations.detail.stop_loss')}</Typography>
                                    <Typography sx={{ fontWeight: 700, fontFamily: 'monospace', color: '#dc2626', fontSize: '1rem' }}>
                                        ¥{safeFixed(stock.stop_loss, 2)}
                                    </Typography>
                                    {price && typeof stock.stop_loss === 'number' && (
                                        <Typography sx={{ color: '#64748b', fontSize: '0.7rem' }}>
                                            {((stock.stop_loss / price - 1) * 100).toFixed(1)}%
                                        </Typography>
                                    )}
                                </Box>
                            )}
                        </Box>
                    )}

                    {!isShortTerm && stock.target_price_1y && (
                        <Box sx={{ bgcolor: '#f8fafc', borderRadius: '8px', p: 1.5 }}>
                            <Typography sx={{ color: '#64748b', fontSize: '0.7rem', mb: 0.5 }}>{t('recommendations.detail.target_price_1y')}</Typography>
                            <Typography sx={{ fontWeight: 700, fontFamily: 'monospace', color: '#1e293b', fontSize: '1rem' }}>
                                ¥{safeFixed(stock.target_price_1y, 2)}
                                <Typography component="span" sx={{ color: '#64748b', fontSize: '0.8rem', ml: 1 }}>
                                    ({stock.expected_return_1y || '-'})
                                </Typography>
                            </Typography>
                        </Box>
                    )}

                    {((stock.key_catalysts && stock.key_catalysts.length > 0) || (stock.risk_factors && stock.risk_factors.length > 0)) && (
                        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
                            {stock.key_catalysts && stock.key_catalysts.length > 0 && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#16a34a', mb: 0.75 }}>{t('recommendations.detail.catalysts')}</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {stock.key_catalysts.map((catalyst, i) => (
                                            <Typography key={i} sx={{ color: '#475569', fontSize: '0.8rem', pl: 1.5, position: 'relative', '&::before': { content: '"•"', position: 'absolute', left: 0, color: '#16a34a' } }}>
                                                {catalyst}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                            {stock.risk_factors && stock.risk_factors.length > 0 && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#dc2626', mb: 0.75 }}>{t('recommendations.detail.risk_factors')}</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {stock.risk_factors.map((risk, i) => (
                                            <Typography key={i} sx={{ color: '#475569', fontSize: '0.8rem', pl: 1.5, position: 'relative', '&::before': { content: '"•"', position: 'absolute', left: 0, color: '#dc2626' } }}>
                                                {risk}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                        </Box>
                    )}

                    {!isShortTerm && (stock.competitive_advantage || stock.valuation_analysis || stock.industry_position) && (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            {stock.competitive_advantage && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.5 }}>{t('recommendations.detail.competitive_advantage')}</Typography>
                                    <Typography sx={{ color: '#475569', fontSize: '0.8rem', lineHeight: 1.5 }}>{stock.competitive_advantage}</Typography>
                                </Box>
                            )}
                            {stock.valuation_analysis && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.5 }}>{t('recommendations.detail.valuation_analysis')}</Typography>
                                    <Typography sx={{ color: '#475569', fontSize: '0.8rem', lineHeight: 1.5 }}>{stock.valuation_analysis}</Typography>
                                </Box>
                            )}
                            {stock.industry_position && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.5 }}>{t('recommendations.detail.industry_position')}</Typography>
                                    <Typography sx={{ color: '#475569', fontSize: '0.8rem', lineHeight: 1.5 }}>{stock.industry_position}</Typography>
                                </Box>
                            )}
                        </Box>
                    )}
                </Box>
            </DialogContent>

            <DialogActions sx={{ px: 2.5, py: 1.5, borderTop: '1px solid #e2e8f0' }}>
                <Typography sx={{ color: '#94a3b8', fontSize: '0.75rem', flex: 1 }}>
                    {t('recommendations.detail.holding_period')}: {stock.holding_period || (isShortTerm ? t('recommendations.detail.holding_short') : t('recommendations.detail.holding_long'))}
                </Typography>
                <Button onClick={onClose} size="small" sx={{ color: '#64748b', fontSize: '0.8rem' }}>
                    {t('recommendations.detail.close')}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

// --- Fund Detail Modal Component ---

interface FundDetailModalProps {
    open: boolean;
    onClose: () => void;
    fund: RecommendationFund | null;
    isShortTerm: boolean;
}

const FundDetailModal = ({ open, onClose, fund, isShortTerm }: FundDetailModalProps) => {
    const { t } = useTranslation();
    if (!fund) return null;

    const formatReturn = (val: number | string | undefined | null): string => {
        if (val === undefined || val === null) return '-';
        const num = typeof val === 'number' ? val : parseFloat(String(val));
        if (isNaN(num)) return '-';
        return `${num > 0 ? '+' : ''}${num.toFixed(2)}%`;
    };

    const getReturnColor = (val: number | string | undefined | null): string => {
        if (val === undefined || val === null) return '#64748b';
        const num = typeof val === 'number' ? val : parseFloat(String(val));
        if (isNaN(num)) return '#64748b';
        return num > 0 ? '#dc2626' : '#16a34a';
    };

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="sm"
            fullWidth
            PaperProps={{ sx: { borderRadius: '12px', maxHeight: '85vh' } }}
        >
            <DialogTitle sx={{ p: 0 }}>
                <Box sx={{ px: 2.5, py: 2, borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                            <Typography sx={{ fontWeight: 700, color: '#1e293b', fontSize: '1.1rem' }}>
                                {fund.name}
                            </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 0.5 }}>
                            <Typography sx={{ color: '#64748b', fontFamily: 'monospace', fontSize: '0.8rem' }}>
                                {fund.code}
                            </Typography>
                            <Chip label={fund.fund_type || t('recommendations.tabs.funds')} size="small" sx={{ height: 18, fontSize: '0.65rem', bgcolor: '#f1f5f9', color: '#475569' }} />
                            <ConfidenceChip confidence={fund.confidence} />
                        </Box>
                    </Box>
                    <IconButton onClick={onClose} size="small" sx={{ color: '#94a3b8' }}>
                        <CloseIcon fontSize="small" />
                    </IconButton>
                </Box>
            </DialogTitle>

            <DialogContent sx={{ p: 0 }}>
                <Box sx={{ px: 2.5, py: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box>
                        <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.75, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <LightbulbIcon sx={{ fontSize: 14 }} /> {t('recommendations.detail.reason')}
                        </Typography>
                        <Typography sx={{ color: '#334155', fontSize: '0.85rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                            {fund.investment_logic || t('recommendations.detail.no_logic')}
                        </Typography>
                    </Box>

                    {fund.why_now && (
                        <Box>
                            <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.75, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <AccessTimeIcon sx={{ fontSize: 14 }} /> {t('recommendations.detail.why_now')}
                            </Typography>
                            <Typography sx={{ color: '#334155', fontSize: '0.85rem', lineHeight: 1.6 }}>
                                {fund.why_now}
                            </Typography>
                        </Box>
                    )}

                    <Box sx={{ bgcolor: '#f8fafc', borderRadius: '8px', p: 1.5 }}>
                        <Typography sx={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', mb: 1 }}>{t('recommendations.detail.fund_data')}</Typography>
                        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1.5 }}>
                            <Box>
                                <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.current_nav')}</Typography>
                                <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                    {typeof fund.current_nav === 'number' ? fund.current_nav.toFixed(4) : '-'}
                                </Typography>
                            </Box>
                            {isShortTerm ? (
                                <>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.return_1w')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: getReturnColor(fund.return_1w), fontSize: '0.85rem' }}>
                                            {formatReturn(fund.return_1w)}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.return_1m')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: getReturnColor(fund.return_1m), fontSize: '0.85rem' }}>
                                            {formatReturn(fund.return_1m)}
                                        </Typography>
                                    </Box>
                                </>
                            ) : (
                                <>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.return_1y')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: getReturnColor(fund.return_1y), fontSize: '0.85rem' }}>
                                            {formatReturn(fund.return_1y)}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.return_3y')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: getReturnColor(fund.return_3y), fontSize: '0.85rem' }}>
                                            {formatReturn(fund.return_3y)}
                                        </Typography>
                                    </Box>
                                </>
                            )}
                            <Box>
                                <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.detail.expected_return')}</Typography>
                                <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                    {fund.expected_return || fund.expected_return_1y || '-'}
                                </Typography>
                            </Box>
                        </Box>
                    </Box>

                    {((fund.key_catalysts && fund.key_catalysts.length > 0) || (fund.risk_factors && fund.risk_factors.length > 0)) && (
                        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
                            {fund.key_catalysts && fund.key_catalysts.length > 0 && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#16a34a', mb: 0.75 }}>{t('recommendations.detail.catalysts')}</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {fund.key_catalysts.map((catalyst, i) => (
                                            <Typography key={i} sx={{ color: '#475569', fontSize: '0.8rem', pl: 1.5, position: 'relative', '&::before': { content: '"•"', position: 'absolute', left: 0, color: '#16a34a' } }}>
                                                {catalyst}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                            {fund.risk_factors && fund.risk_factors.length > 0 && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#dc2626', mb: 0.75 }}>{t('recommendations.detail.risk_factors')}</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {fund.risk_factors.map((risk, i) => (
                                            <Typography key={i} sx={{ color: '#475569', fontSize: '0.8rem', pl: 1.5, position: 'relative', '&::before': { content: '"•"', position: 'absolute', left: 0, color: '#dc2626' } }}>
                                                {risk}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                        </Box>
                    )}

                    {!isShortTerm && (fund.manager_analysis || fund.fund_style || fund.suitable_for) && (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            {fund.manager_analysis && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.5 }}>{t('recommendations.detail.manager_analysis')}</Typography>
                                    <Typography sx={{ color: '#475569', fontSize: '0.8rem', lineHeight: 1.5 }}>{fund.manager_analysis}</Typography>
                                </Box>
                            )}
                            {fund.fund_style && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.5 }}>{t('recommendations.detail.fund_style')}</Typography>
                                    <Typography sx={{ color: '#475569', fontSize: '0.8rem', lineHeight: 1.5 }}>{fund.fund_style}</Typography>
                                </Box>
                            )}
                            {fund.suitable_for && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.5 }}>{t('recommendations.detail.suitable_for')}</Typography>
                                    <Typography sx={{ color: '#475569', fontSize: '0.8rem', lineHeight: 1.5 }}>{fund.suitable_for}</Typography>
                                </Box>
                            )}
                        </Box>
                    )}
                </Box>
            </DialogContent>

            <DialogActions sx={{ px: 2.5, py: 1.5, borderTop: '1px solid #e2e8f0' }}>
                <Typography sx={{ color: '#94a3b8', fontSize: '0.75rem', flex: 1 }}>
                    {t('recommendations.detail.holding_period')}: {fund.holding_period || (isShortTerm ? t('recommendations.detail.holding_short') : t('recommendations.detail.holding_long'))}
                </Typography>
                <Button onClick={onClose} size="small" sx={{ color: '#64748b', fontSize: '0.8rem' }}>
                    {t('recommendations.detail.close')}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

// --- Stock Table Component ---

interface StockTableProps {
    stocks: RecommendationStock[];
    isShortTerm: boolean;
    onStockClick?: (stock: RecommendationStock) => void;
}

const StockTable = ({ stocks, isShortTerm, onStockClick }: StockTableProps) => {
    const { t } = useTranslation();

    if (!stocks || stocks.length === 0) {
        return <Typography className="text-slate-400 text-center py-8">{t('recommendations.no_data')}</Typography>;
    }

    return (
        <TableContainer>
            <Table size="small">
                <TableHead>
                    <TableRow className="bg-slate-50">
                        <TableCell className="font-bold text-slate-600 text-xs w-8">{t('recommendations.table.rank')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.code')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.name')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.price')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.change')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs w-28">{t('recommendations.table.score')}</TableCell>
                        {isShortTerm ? (
                            <>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.net_inflow')}</TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.volume_ratio')}</TableCell>
                            </>
                        ) : (
                            <>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.pe')}</TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.market_cap')}</TableCell>
                            </>
                        )}
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.confidence')}</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {stocks.map((stock, index) => (
                        <TableRow
                            key={stock.code}
                            className="hover:bg-blue-50 transition-colors cursor-pointer"
                            onClick={() => onStockClick?.(stock)}
                        >
                            <TableCell>
                                <Box className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                    index < 3 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'
                                }`}>
                                    {index + 1}
                                </Box>
                            </TableCell>
                            <TableCell>
                                <NumberMono className="text-sm font-semibold text-slate-700">{stock.code}</NumberMono>
                            </TableCell>
                            <TableCell>
                                <Typography className="text-sm font-medium text-slate-800 truncate max-w-[120px]">
                                    {stock.name}
                                </Typography>
                            </TableCell>
                            <TableCell className="text-right">
                                <NumberMono className="text-sm font-semibold text-slate-800">
                                    {(stock.current_price || stock.price)?.toFixed(2) || '-'}
                                </NumberMono>
                            </TableCell>
                            <TableCell className="text-right">
                                <ColorVal val={stock.change_pct} suffix="%" />
                            </TableCell>
                            <TableCell>
                                <ScoreBar score={stock.recommendation_score || stock.score || 0} />
                            </TableCell>
                            {isShortTerm ? (
                                <>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${(stock.main_net_inflow || 0) > 0 ? 'text-red-600' : 'text-green-600'}`}>
                                            {formatAmount(stock.main_net_inflow)}
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className="text-sm text-slate-600">
                                            {stock.volume_ratio?.toFixed(2) || '-'}
                                        </NumberMono>
                                    </TableCell>
                                </>
                            ) : (
                                <>
                                    <TableCell className="text-right">
                                        <NumberMono className="text-sm text-slate-600">
                                            {stock.pe?.toFixed(1) || '-'}
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className="text-sm text-slate-600">
                                            {formatMarketCap(stock.market_cap)}
                                        </NumberMono>
                                    </TableCell>
                                </>
                            )}
                            <TableCell>
                                <ConfidenceChip confidence={stock.confidence} />
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

// --- Fund Table Component ---

interface FundTableProps {
    funds: RecommendationFund[];
    isShortTerm: boolean;
    onFundClick?: (fund: RecommendationFund) => void;
}

// --- V2 Stock Table Component (with factor display) ---

interface StockTableV2Props {
    stocks: RecommendationStockV2[];
    isShortTerm: boolean;
    onStockClick?: (stock: RecommendationStockV2) => void;
}

const StockTableV2 = ({ stocks, isShortTerm, onStockClick }: StockTableV2Props) => {
    const { t } = useTranslation();

    if (!stocks || stocks.length === 0) {
        return <Typography className="text-slate-400 text-center py-8">{t('recommendations.no_data')}</Typography>;
    }

    return (
        <TableContainer>
            <Table size="small">
                <TableHead>
                    <TableRow className="bg-slate-50">
                        <TableCell className="font-bold text-slate-600 text-xs w-8">{t('recommendations.table.rank')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.code')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.name')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs w-28">{t('recommendations.table.score')}</TableCell>
                        {isShortTerm ? (
                            <>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.consolidation_score_tip')}>
                                        <span>{t('recommendations.v2.consolidation')}</span>
                                    </Tooltip>
                                </TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.volume_precursor_tip')}>
                                        <span>{t('recommendations.v2.volume')}</span>
                                    </Tooltip>
                                </TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.main_inflow_tip')}>
                                        <span>{t('recommendations.v2.inflow_5d')}</span>
                                    </Tooltip>
                                </TableCell>
                            </>
                        ) : (
                            <>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.roe_tip')}>
                                        <span>ROE</span>
                                    </Tooltip>
                                </TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.peg_tip')}>
                                        <span>PEG</span>
                                    </Tooltip>
                                </TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.quality_score_tip')}>
                                        <span>{t('recommendations.v2.quality')}</span>
                                    </Tooltip>
                                </TableCell>
                            </>
                        )}
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.v2.strategy')}</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {stocks.map((stock, index) => (
                        <TableRow
                            key={stock.code}
                            className="hover:bg-blue-50 transition-colors cursor-pointer"
                            onClick={() => onStockClick?.(stock)}
                        >
                            <TableCell>
                                <Box className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                    index < 3 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'
                                }`}>
                                    {index + 1}
                                </Box>
                            </TableCell>
                            <TableCell>
                                <NumberMono className="text-sm font-semibold text-slate-700">{stock.code}</NumberMono>
                            </TableCell>
                            <TableCell>
                                <Box>
                                    <Typography className="text-sm font-medium text-slate-800 truncate max-w-[120px]">
                                        {stock.name}
                                    </Typography>
                                    {stock.industry && (
                                        <Typography className="text-[10px] text-slate-400">{stock.industry}</Typography>
                                    )}
                                </Box>
                            </TableCell>
                            <TableCell>
                                <ScoreBar score={stock.score} />
                            </TableCell>
                            {isShortTerm ? (
                                <>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (stock.factors?.consolidation_score || 0) >= 70 ? 'text-green-600 font-semibold' : 'text-slate-600'
                                        }`}>
                                            {stock.factors?.consolidation_score?.toFixed(0) || '-'}
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (stock.factors?.volume_precursor || 0) >= 60 ? 'text-blue-600 font-semibold' : 'text-slate-600'
                                        }`}>
                                            {stock.factors?.volume_precursor?.toFixed(0) || '-'}
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (stock.factors?.main_inflow_5d || 0) > 0 ? 'text-red-600' : 'text-green-600'
                                        }`}>
                                            {formatAmount(stock.factors?.main_inflow_5d)}
                                        </NumberMono>
                                    </TableCell>
                                </>
                            ) : (
                                <>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (stock.factors?.roe || 0) >= 15 ? 'text-green-600 font-semibold' : 'text-slate-600'
                                        }`}>
                                            {stock.factors?.roe?.toFixed(1) || '-'}%
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (stock.factors?.peg_ratio || 0) > 0 && (stock.factors?.peg_ratio || 0) < 1
                                                ? 'text-green-600 font-semibold'
                                                : 'text-slate-600'
                                        }`}>
                                            {stock.factors?.peg_ratio?.toFixed(2) || '-'}
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (stock.factors?.quality_score || 0) >= 70 ? 'text-green-600 font-semibold' : 'text-slate-600'
                                        }`}>
                                            {stock.factors?.quality_score?.toFixed(0) || '-'}
                                        </NumberMono>
                                    </TableCell>
                                </>
                            )}
                            <TableCell>
                                <Chip
                                    label={stock.strategy || (isShortTerm ? t('recommendations.v2.breakout') : t('recommendations.v2.quality'))}
                                    size="small"
                                    className={`h-5 text-[10px] font-medium ${
                                        isShortTerm ? 'bg-blue-50 text-blue-700' : 'bg-purple-50 text-purple-700'
                                    }`}
                                />
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

// --- V2 Fund Table Component (with factor display) ---

interface FundTableV2Props {
    funds: RecommendationFundV2[];
    isShortTerm: boolean;
    onFundClick?: (fund: RecommendationFundV2) => void;
}

const FundTableV2 = ({ funds, isShortTerm, onFundClick }: FundTableV2Props) => {
    const { t } = useTranslation();

    if (!funds || funds.length === 0) {
        return <Typography className="text-slate-400 text-center py-8">{t('recommendations.no_data')}</Typography>;
    }

    return (
        <TableContainer>
            <Table size="small">
                <TableHead>
                    <TableRow className="bg-slate-50">
                        <TableCell className="font-bold text-slate-600 text-xs w-8">{t('recommendations.table.rank')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.code')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.name')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs w-28">{t('recommendations.table.score')}</TableCell>
                        {isShortTerm ? (
                            <>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.sharpe_tip')}>
                                        <span>{t('recommendations.v2.sharpe_20d')}</span>
                                    </Tooltip>
                                </TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.return_1m_tip')}>
                                        <span>{t('recommendations.table.return_1m')}</span>
                                    </Tooltip>
                                </TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.volatility_tip')}>
                                        <span>{t('recommendations.v2.volatility')}</span>
                                    </Tooltip>
                                </TableCell>
                            </>
                        ) : (
                            <>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.sharpe_1y_tip')}>
                                        <span>{t('recommendations.v2.sharpe_1y')}</span>
                                    </Tooltip>
                                </TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.max_dd_tip')}>
                                        <span>{t('recommendations.v2.max_dd')}</span>
                                    </Tooltip>
                                </TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">
                                    <Tooltip title={t('recommendations.v2.manager_tenure_tip')}>
                                        <span>{t('recommendations.v2.tenure')}</span>
                                    </Tooltip>
                                </TableCell>
                            </>
                        )}
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.v2.strategy')}</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {funds.map((fund, index) => (
                        <TableRow
                            key={fund.code}
                            className="hover:bg-purple-50 transition-colors cursor-pointer"
                            onClick={() => onFundClick?.(fund)}
                        >
                            <TableCell>
                                <Box className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                    index < 3 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'
                                }`}>
                                    {index + 1}
                                </Box>
                            </TableCell>
                            <TableCell>
                                <NumberMono className="text-sm font-semibold text-slate-700">{fund.code}</NumberMono>
                            </TableCell>
                            <TableCell>
                                <Box>
                                    <Typography className="text-sm font-medium text-slate-800 truncate max-w-[160px]">
                                        {fund.name}
                                    </Typography>
                                    {fund.type && (
                                        <Typography className="text-[10px] text-slate-400">{fund.type}</Typography>
                                    )}
                                </Box>
                            </TableCell>
                            <TableCell>
                                <ScoreBar score={fund.score} />
                            </TableCell>
                            {isShortTerm ? (
                                <>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (fund.factors?.sharpe_20d || 0) >= 1.5 ? 'text-green-600 font-semibold' : 'text-slate-600'
                                        }`}>
                                            {fund.factors?.sharpe_20d?.toFixed(2) || '-'}
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <ColorVal val={fund.factors?.return_1m} suffix="%" />
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (fund.factors?.volatility_60d || 0) < 15 ? 'text-green-600' : 'text-orange-600'
                                        }`}>
                                            {fund.factors?.volatility_60d?.toFixed(1) || '-'}%
                                        </NumberMono>
                                    </TableCell>
                                </>
                            ) : (
                                <>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (fund.factors?.sharpe_1y || 0) >= 1.0 ? 'text-green-600 font-semibold' : 'text-slate-600'
                                        }`}>
                                            {fund.factors?.sharpe_1y?.toFixed(2) || '-'}
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            Math.abs(fund.factors?.max_drawdown_1y || 0) < 15 ? 'text-green-600' : 'text-red-600'
                                        }`}>
                                            {fund.factors?.max_drawdown_1y?.toFixed(1) || '-'}%
                                        </NumberMono>
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <NumberMono className={`text-sm ${
                                            (fund.factors?.manager_tenure_years || 0) >= 3 ? 'text-green-600 font-semibold' : 'text-slate-600'
                                        }`}>
                                            {fund.factors?.manager_tenure_years?.toFixed(1) || '-'}{t('common.years')}
                                        </NumberMono>
                                    </TableCell>
                                </>
                            )}
                            <TableCell>
                                <Chip
                                    label={fund.strategy || (isShortTerm ? t('recommendations.v2.momentum') : t('recommendations.v2.alpha'))}
                                    size="small"
                                    className={`h-5 text-[10px] font-medium ${
                                        isShortTerm ? 'bg-blue-50 text-blue-700' : 'bg-purple-50 text-purple-700'
                                    }`}
                                />
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

const FundTable = ({ funds, isShortTerm, onFundClick }: FundTableProps) => {
    const { t } = useTranslation();

    if (!funds || funds.length === 0) {
        return <Typography className="text-slate-400 text-center py-8">{t('recommendations.no_data')}</Typography>;
    }

    return (
        <TableContainer>
            <Table size="small">
                <TableHead>
                    <TableRow className="bg-slate-50">
                        <TableCell className="font-bold text-slate-600 text-xs w-8">{t('recommendations.table.rank')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.code')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.name')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.fund_type')}</TableCell>
                        {isShortTerm ? (
                            <>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.return_1w')}</TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.return_1m')}</TableCell>
                            </>
                        ) : (
                            <>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.return_1y')}</TableCell>
                                <TableCell className="font-bold text-slate-600 text-xs text-right">{t('recommendations.table.return_3y')}</TableCell>
                            </>
                        )}
                        <TableCell className="font-bold text-slate-600 text-xs w-28">{t('recommendations.table.score')}</TableCell>
                        <TableCell className="font-bold text-slate-600 text-xs">{t('recommendations.table.confidence')}</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {funds.map((fund, index) => (
                        <TableRow
                            key={fund.code}
                            className="hover:bg-purple-50 transition-colors cursor-pointer"
                            onClick={() => onFundClick?.(fund)}
                        >
                            <TableCell>
                                <Box className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                    index < 3 ? 'bg-amber-100 text-amber-700' : 'bg-slate-100 text-slate-600'
                                }`}>
                                    {index + 1}
                                </Box>
                            </TableCell>
                            <TableCell>
                                <NumberMono className="text-sm font-semibold text-slate-700">{fund.code}</NumberMono>
                            </TableCell>
                            <TableCell>
                                <Typography className="text-sm font-medium text-slate-800 truncate max-w-[160px]">
                                    {fund.name}
                                </Typography>
                            </TableCell>
                            <TableCell>
                                <Chip
                                    label={fund.fund_type || '-'}
                                    size="small"
                                    className="h-5 text-[10px] bg-indigo-50 text-indigo-700 font-medium"
                                />
                            </TableCell>
                            {isShortTerm ? (
                                <>
                                    <TableCell className="text-right">
                                        <ColorVal val={fund.return_1w}  />
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <ColorVal val={fund.return_1m} />
                                    </TableCell>
                                </>
                            ) : (
                                <>
                                    <TableCell className="text-right">
                                        <ColorVal val={fund.return_1y}  />
                                    </TableCell>
                                    <TableCell className="text-right">
                                        <ColorVal val={fund.return_3y}  />
                                    </TableCell>
                                </>
                            )}
                            <TableCell>
                                <ScoreBar score={fund.recommendation_score || fund.score || 0} />
                            </TableCell>
                            <TableCell>
                                <ConfidenceChip confidence={fund.confidence} />
                            </TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    );
};

// --- Recommendation Section Component ---

interface RecommendationSectionProps {
    title: string;
    subtitle: string;
    icon: React.ReactNode;
    stocks: RecommendationStock[];
    funds: RecommendationFund[];
    marketView?: string;
    sectorPreference?: string[];
    riskWarning?: string;
    isShortTerm: boolean;
    defaultExpanded?: boolean;
}

const RecommendationSection = ({
    title,
    subtitle,
    icon,
    stocks,
    funds,
    marketView,
    sectorPreference,
    riskWarning,
    isShortTerm,
    defaultExpanded = true,
}: RecommendationSectionProps) => {
    const { t } = useTranslation();
    const [tabValue, setTabValue] = useState(0);
    const [expanded, setExpanded] = useState(defaultExpanded);

    // Detail modal states
    const [selectedStock, setSelectedStock] = useState<RecommendationStock | null>(null);
    const [selectedFund, setSelectedFund] = useState<RecommendationFund | null>(null);
    const [stockModalOpen, setStockModalOpen] = useState(false);
    const [fundModalOpen, setFundModalOpen] = useState(false);

    // Hover preview state
    const [previewAnchorEl, setPreviewAnchorEl] = useState<HTMLElement | null>(null);
    const [previewType, setPreviewType] = useState<'stocks' | 'funds'>('stocks');
    const [showPreview, setShowPreview] = useState(false);
    const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const isHoveringRef = useRef(false);

    // Clear timer on unmount
    useEffect(() => {
        return () => {
            if (hoverTimerRef.current) {
                clearTimeout(hoverTimerRef.current);
            }
        };
    }, []);

    const handleStockClick = useCallback((stock: RecommendationStock) => {
        setSelectedStock(stock);
        setStockModalOpen(true);
    }, []);

    const handleFundClick = useCallback((fund: RecommendationFund) => {
        setSelectedFund(fund);
        setFundModalOpen(true);
    }, []);

    const handleTabMouseEnter = useCallback((event: React.MouseEvent<HTMLElement>, type: 'stocks' | 'funds', isActive: boolean) => {
        // Only show preview for non-active tabs
        if (isActive) return;

        isHoveringRef.current = true;
        setPreviewAnchorEl(event.currentTarget);
        setPreviewType(type);

        // Delay showing preview by 300ms
        hoverTimerRef.current = setTimeout(() => {
            if (isHoveringRef.current) {
                setShowPreview(true);
            }
        }, 300);
    }, []);

    const handleTabMouseLeave = useCallback(() => {
        isHoveringRef.current = false;
        if (hoverTimerRef.current) {
            clearTimeout(hoverTimerRef.current);
            hoverTimerRef.current = null;
        }
        setShowPreview(false);
    }, []);

    const handlePopoverClose = useCallback(() => {
        setShowPreview(false);
    }, []);

    return (
        <Paper elevation={0} className="border border-slate-200 rounded-xl bg-white overflow-hidden shadow-sm">
            {/* Section Header */}
            <Box
                className="px-5 py-4 flex items-center justify-between cursor-pointer hover:bg-slate-50 transition-colors"
                onClick={() => setExpanded(!expanded)}
            >
                <Box className="flex items-center gap-3">
                    <Box className={`w-10 h-10 rounded-lg flex items-center justify-center ${isShortTerm ? 'bg-blue-100' : 'bg-purple-100'}`}>
                        {icon}
                    </Box>
                    <Box>
                        <Typography variant="h6" className="font-bold text-slate-800">{title}</Typography>
                        <Typography variant="caption" className="text-slate-500">{subtitle}</Typography>
                    </Box>
                </Box>
                <Box className="flex items-center gap-2">
                    <Chip
                        label={`${stocks?.length || 0} ${t('recommendations.tabs.stocks')}`}
                        size="small"
                        className="h-6 text-xs bg-slate-100 text-slate-600"
                    />
                    <Chip
                        label={`${funds?.length || 0} ${t('recommendations.tabs.funds')}`}
                        size="small"
                        className="h-6 text-xs bg-slate-100 text-slate-600"
                    />
                    <IconButton size="small">
                        {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                </Box>
            </Box>

            <Collapse in={expanded}>
                {/* Market View & Sector Cards */}
                {(marketView || (sectorPreference && sectorPreference.length > 0)) && (
                    <Box className="px-5 pb-3 flex flex-wrap gap-3">
                        {marketView && (
                            <Box className="flex-1 min-w-[200px] p-3 bg-slate-50 rounded-lg border border-slate-100">
                                <Box className="flex items-center gap-2 mb-2">
                                    <InsightsIcon className="text-slate-400 text-sm" />
                                    <Typography variant="caption" className="text-slate-500 font-bold uppercase">
                                        {t('recommendations.market_view.title')}
                                    </Typography>
                                </Box>
                                <Typography variant="body2" className="text-slate-700">
                                    {marketView}
                                </Typography>
                            </Box>
                        )}
                        {sectorPreference && sectorPreference.length > 0 && (
                            <Box className="flex-1 min-w-[200px] p-3 bg-slate-50 rounded-lg border border-slate-100">
                                <Box className="flex items-center gap-2 mb-2">
                                    <TrendingUpIcon className="text-slate-400 text-sm" />
                                    <Typography variant="caption" className="text-slate-500 font-bold uppercase">
                                        {t('recommendations.sector.hot_sectors')}
                                    </Typography>
                                </Box>
                                <Box className="flex flex-wrap gap-1">
                                    {sectorPreference.map((sector, i) => (
                                        <Chip key={i} label={sector} size="small" className="h-6 text-xs bg-green-50 text-green-700" />
                                    ))}
                                </Box>
                            </Box>
                        )}
                    </Box>
                )}

                {/* Tabs for Stocks/Funds */}
                <Box className="border-t border-slate-100">
                    <Tabs
                        value={tabValue}
                        onChange={(_, v) => setTabValue(v)}
                        className="px-5"
                        TabIndicatorProps={{ className: 'bg-blue-600' }}
                    >
                        <Tab
                            icon={<ShowChartIcon className="text-sm" />}
                            iconPosition="start"
                            label={t('recommendations.tabs.stocks')}
                            className="min-h-[48px] text-sm"
                            onMouseEnter={(e) => handleTabMouseEnter(e, 'stocks', tabValue === 0)}
                            onMouseLeave={handleTabMouseLeave}
                        />
                        <Tab
                            icon={<PieChartIcon className="text-sm" />}
                            iconPosition="start"
                            label={t('recommendations.tabs.funds')}
                            className="min-h-[48px] text-sm"
                            onMouseEnter={(e) => handleTabMouseEnter(e, 'funds', tabValue === 1)}
                            onMouseLeave={handleTabMouseLeave}
                        />
                    </Tabs>

                    {/* Hover Preview Popover */}
                    <Popover
                        open={showPreview}
                        anchorEl={previewAnchorEl}
                        onClose={handlePopoverClose}
                        anchorOrigin={{
                            vertical: 'bottom',
                            horizontal: 'center',
                        }}
                        transformOrigin={{
                            vertical: 'top',
                            horizontal: 'center',
                        }}
                        disableRestoreFocus
                        sx={{
                            pointerEvents: 'none',
                            '& .MuiPopover-paper': {
                                pointerEvents: 'auto',
                                borderRadius: '16px',
                                boxShadow: '0 12px 40px rgba(0,0,0,0.15)',
                                border: 'none',
                                mt: 1,
                                overflow: 'hidden',
                            }
                        }}
                        TransitionComponent={Fade}
                        TransitionProps={{ timeout: 200 }}
                    >
                        <TabPreviewContent
                            type={previewType}
                            stocks={stocks}
                            funds={funds}
                            isShortTerm={isShortTerm}
                        />
                    </Popover>
                </Box>

                {/* Table Content */}
                <Box className="px-5 pb-5">
                    {tabValue === 0 ? (
                        <StockTable stocks={stocks} isShortTerm={isShortTerm} onStockClick={handleStockClick} />
                    ) : (
                        <FundTable funds={funds} isShortTerm={isShortTerm} onFundClick={handleFundClick} />
                    )}
                </Box>

                {/* Risk Warning */}
                {riskWarning && (
                    <Box className="px-5 pb-4">
                        <Alert
                            severity="warning"
                            icon={<WarningAmberIcon className="text-amber-600" />}
                            className="bg-amber-50 border border-amber-200"
                        >
                            <Typography variant="body2" className="text-amber-800">
                                {riskWarning}
                            </Typography>
                        </Alert>
                    </Box>
                )}
            </Collapse>

            {/* Stock Detail Modal */}
            <StockDetailModal
                open={stockModalOpen}
                onClose={() => setStockModalOpen(false)}
                stock={selectedStock}
                isShortTerm={isShortTerm}
            />

            {/* Fund Detail Modal */}
            <FundDetailModal
                open={fundModalOpen}
                onClose={() => setFundModalOpen(false)}
                fund={selectedFund}
                isShortTerm={isShortTerm}
            />
        </Paper>
    );
};

// --- V2 Recommendation Section Component ---

interface RecommendationSectionV2Props {
    title: string;
    subtitle: string;
    icon: React.ReactNode;
    stocks: RecommendationStockV2[];
    funds: RecommendationFundV2[];
    marketView?: string;
    isShortTerm: boolean;
    defaultExpanded?: boolean;
}

const RecommendationSectionV2 = ({
    title,
    subtitle,
    icon,
    stocks,
    funds,
    marketView,
    isShortTerm,
    defaultExpanded = true,
}: RecommendationSectionV2Props) => {
    const { t } = useTranslation();
    const [tabValue, setTabValue] = useState(0);
    const [expanded, setExpanded] = useState(defaultExpanded);

    // Detail modal states for V2
    const [selectedStock, setSelectedStock] = useState<RecommendationStockV2 | null>(null);
    const [selectedFund, setSelectedFund] = useState<RecommendationFundV2 | null>(null);
    const [stockModalOpen, setStockModalOpen] = useState(false);
    const [fundModalOpen, setFundModalOpen] = useState(false);

    const handleStockClick = useCallback((stock: RecommendationStockV2) => {
        setSelectedStock(stock);
        setStockModalOpen(true);
    }, []);

    const handleFundClick = useCallback((fund: RecommendationFundV2) => {
        setSelectedFund(fund);
        setFundModalOpen(true);
    }, []);

    return (
        <Paper elevation={0} className="border border-slate-200 rounded-xl bg-white overflow-hidden shadow-sm">
            {/* Section Header */}
            <Box
                className="px-5 py-4 flex items-center justify-between cursor-pointer hover:bg-slate-50 transition-colors"
                onClick={() => setExpanded(!expanded)}
            >
                <Box className="flex items-center gap-3">
                    <Box className={`w-10 h-10 rounded-lg flex items-center justify-center ${isShortTerm ? 'bg-blue-100' : 'bg-purple-100'}`}>
                        {icon}
                    </Box>
                    <Box>
                        <Box className="flex items-center gap-2">
                            <Typography variant="h6" className="font-bold text-slate-800">{title}</Typography>
                            <Chip
                                label="V2"
                                size="small"
                                className="h-4 text-[9px] font-bold bg-emerald-100 text-emerald-700"
                            />
                        </Box>
                        <Typography variant="caption" className="text-slate-500">{subtitle}</Typography>
                    </Box>
                </Box>
                <Box className="flex items-center gap-2">
                    <Chip
                        label={`${stocks?.length || 0} ${t('recommendations.tabs.stocks')}`}
                        size="small"
                        className="h-6 text-xs bg-slate-100 text-slate-600"
                    />
                    <Chip
                        label={`${funds?.length || 0} ${t('recommendations.tabs.funds')}`}
                        size="small"
                        className="h-6 text-xs bg-slate-100 text-slate-600"
                    />
                    <IconButton size="small">
                        {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                </Box>
            </Box>

            <Collapse in={expanded}>
                {/* Market View Card */}
                {marketView && (
                    <Box className="px-5 pb-3">
                        <Box className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                            <Box className="flex items-center gap-2 mb-2">
                                <InsightsIcon className="text-slate-400 text-sm" />
                                <Typography variant="caption" className="text-slate-500 font-bold uppercase">
                                    {isShortTerm ? t('recommendations.market_view.title') : t('recommendations.v2.macro_view')}
                                </Typography>
                            </Box>
                            <Typography variant="body2" className="text-slate-700">
                                {marketView}
                            </Typography>
                        </Box>
                    </Box>
                )}

                {/* Tabs for Stocks/Funds */}
                <Box className="border-t border-slate-100">
                    <Tabs
                        value={tabValue}
                        onChange={(_, v) => setTabValue(v)}
                        className="px-5"
                        TabIndicatorProps={{ className: 'bg-emerald-600' }}
                    >
                        <Tab
                            icon={<ShowChartIcon className="text-sm" />}
                            iconPosition="start"
                            label={t('recommendations.tabs.stocks')}
                            className="min-h-[48px] text-sm"
                        />
                        <Tab
                            icon={<PieChartIcon className="text-sm" />}
                            iconPosition="start"
                            label={t('recommendations.tabs.funds')}
                            className="min-h-[48px] text-sm"
                        />
                    </Tabs>
                </Box>

                {/* Table Content */}
                <Box className="px-5 pb-5">
                    {tabValue === 0 ? (
                        <StockTableV2 stocks={stocks} isShortTerm={isShortTerm} onStockClick={handleStockClick} />
                    ) : (
                        <FundTableV2 funds={funds} isShortTerm={isShortTerm} onFundClick={handleFundClick} />
                    )}
                </Box>
            </Collapse>

            {/* V2 Stock Detail Modal */}
            <StockDetailModalV2
                open={stockModalOpen}
                onClose={() => setStockModalOpen(false)}
                stock={selectedStock}
                isShortTerm={isShortTerm}
            />

            {/* V2 Fund Detail Modal */}
            <FundDetailModalV2
                open={fundModalOpen}
                onClose={() => setFundModalOpen(false)}
                fund={selectedFund}
                isShortTerm={isShortTerm}
            />
        </Paper>
    );
};

// --- V2 Stock Detail Modal Component ---

interface StockDetailModalV2Props {
    open: boolean;
    onClose: () => void;
    stock: RecommendationStockV2 | null;
    isShortTerm: boolean;
}

const StockDetailModalV2 = ({ open, onClose, stock, isShortTerm }: StockDetailModalV2Props) => {
    const { t } = useTranslation();
    if (!stock) return null;

    const factors = stock.factors || {};

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="sm"
            fullWidth
            PaperProps={{ sx: { borderRadius: '12px', maxHeight: '85vh' } }}
        >
            <DialogTitle sx={{ p: 0 }}>
                <Box sx={{ px: 2.5, py: 2, borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                            <Typography sx={{ fontWeight: 700, color: '#1e293b', fontSize: '1.1rem' }}>
                                {stock.name}
                            </Typography>
                            <Typography sx={{ color: '#64748b', fontFamily: 'monospace', fontSize: '0.8rem' }}>
                                {stock.code}
                            </Typography>
                            <Chip label="V2" size="small" sx={{ height: 16, fontSize: '0.6rem', bgcolor: '#d1fae5', color: '#047857' }} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 0.5 }}>
                            {stock.industry && (
                                <Chip label={stock.industry} size="small" sx={{ height: 18, fontSize: '0.65rem', bgcolor: '#f1f5f9', color: '#475569' }} />
                            )}
                            <Chip
                                label={`${t('recommendations.table.score')}: ${stock.score.toFixed(0)}`}
                                size="small"
                                sx={{ height: 18, fontSize: '0.65rem', bgcolor: '#dbeafe', color: '#1d4ed8', fontWeight: 600 }}
                            />
                        </Box>
                    </Box>
                    <IconButton onClick={onClose} size="small" sx={{ color: '#94a3b8' }}>
                        <CloseIcon fontSize="small" />
                    </IconButton>
                </Box>
            </DialogTitle>

            <DialogContent sx={{ p: 0 }}>
                <Box sx={{ px: 2.5, py: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {/* Explanation */}
                    {stock.explanation && (
                        <Box>
                            <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.75, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <LightbulbIcon sx={{ fontSize: 14 }} /> {t('recommendations.detail.reason')}
                            </Typography>
                            <Typography sx={{ color: '#334155', fontSize: '0.85rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                                {stock.explanation}
                            </Typography>
                        </Box>
                    )}

                    {/* Factor Data Grid */}
                    <Box sx={{ bgcolor: '#f8fafc', borderRadius: '8px', p: 1.5 }}>
                        <Typography sx={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', mb: 1 }}>
                            {isShortTerm ? t('recommendations.v2.technical_factors') : t('recommendations.v2.fundamental_factors')}
                        </Typography>
                        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1.5 }}>
                            {isShortTerm ? (
                                <>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.consolidation')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.consolidation_score || 0) >= 70 ? '#16a34a' : '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.consolidation_score?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.volume')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.volume_precursor?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.ma_conv')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.ma_convergence?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.inflow_5d')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.main_inflow_5d || 0) > 0 ? '#dc2626' : '#16a34a', fontSize: '0.85rem' }}>
                                            {formatAmount(factors.main_inflow_5d)}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>RSI</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.rsi?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.accumulation')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: factors.is_accumulation ? '#16a34a' : '#64748b', fontSize: '0.85rem' }}>
                                            {factors.is_accumulation ? t('common.yes') : t('common.no')}
                                        </Typography>
                                    </Box>
                                </>
                            ) : (
                                <>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>ROE</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.roe || 0) >= 15 ? '#16a34a' : '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.roe?.toFixed(1) || '-'}%
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>PEG</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.peg_ratio || 0) > 0 && (factors.peg_ratio || 0) < 1 ? '#16a34a' : '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.peg_ratio?.toFixed(2) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.quality')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.quality_score || 0) >= 70 ? '#16a34a' : '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.quality_score?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.growth')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.growth_score?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.valuation')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.valuation_score?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.gross_margin')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.gross_margin?.toFixed(1) || '-'}%
                                        </Typography>
                                    </Box>
                                </>
                            )}
                        </Box>
                    </Box>

                    {/* Catalysts & Risks */}
                    {((stock.catalysts && stock.catalysts.length > 0) || (stock.risks && stock.risks.length > 0)) && (
                        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
                            {stock.catalysts && stock.catalysts.length > 0 && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#16a34a', mb: 0.75 }}>{t('recommendations.detail.catalysts')}</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {stock.catalysts.map((catalyst, i) => (
                                            <Typography key={i} sx={{ color: '#475569', fontSize: '0.8rem', pl: 1.5, position: 'relative', '&::before': { content: '"•"', position: 'absolute', left: 0, color: '#16a34a' } }}>
                                                {catalyst}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                            {stock.risks && stock.risks.length > 0 && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#dc2626', mb: 0.75 }}>{t('recommendations.detail.risk_factors')}</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {stock.risks.map((risk, i) => (
                                            <Typography key={i} sx={{ color: '#475569', fontSize: '0.8rem', pl: 1.5, position: 'relative', '&::before': { content: '"•"', position: 'absolute', left: 0, color: '#dc2626' } }}>
                                                {risk}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                        </Box>
                    )}
                </Box>
            </DialogContent>

            <DialogActions sx={{ px: 2.5, py: 1.5, borderTop: '1px solid #e2e8f0' }}>
                <Typography sx={{ color: '#94a3b8', fontSize: '0.75rem', flex: 1 }}>
                    {t('recommendations.v2.strategy')}: {stock.strategy || (isShortTerm ? t('recommendations.v2.breakout') : t('recommendations.v2.quality'))}
                </Typography>
                <Button onClick={onClose} size="small" sx={{ color: '#64748b', fontSize: '0.8rem' }}>
                    {t('recommendations.detail.close')}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

// --- V2 Fund Detail Modal Component ---

interface FundDetailModalV2Props {
    open: boolean;
    onClose: () => void;
    fund: RecommendationFundV2 | null;
    isShortTerm: boolean;
}

const FundDetailModalV2 = ({ open, onClose, fund, isShortTerm }: FundDetailModalV2Props) => {
    const { t } = useTranslation();
    if (!fund) return null;

    const factors = fund.factors || {};

    return (
        <Dialog
            open={open}
            onClose={onClose}
            maxWidth="sm"
            fullWidth
            PaperProps={{ sx: { borderRadius: '12px', maxHeight: '85vh' } }}
        >
            <DialogTitle sx={{ p: 0 }}>
                <Box sx={{ px: 2.5, py: 2, borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                            <Typography sx={{ fontWeight: 700, color: '#1e293b', fontSize: '1.1rem' }}>
                                {fund.name}
                            </Typography>
                            <Chip label="V2" size="small" sx={{ height: 16, fontSize: '0.6rem', bgcolor: '#d1fae5', color: '#047857' }} />
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mt: 0.5 }}>
                            <Typography sx={{ color: '#64748b', fontFamily: 'monospace', fontSize: '0.8rem' }}>
                                {fund.code}
                            </Typography>
                            {fund.type && (
                                <Chip label={fund.type} size="small" sx={{ height: 18, fontSize: '0.65rem', bgcolor: '#f1f5f9', color: '#475569' }} />
                            )}
                            <Chip
                                label={`${t('recommendations.table.score')}: ${fund.score.toFixed(0)}`}
                                size="small"
                                sx={{ height: 18, fontSize: '0.65rem', bgcolor: '#e0e7ff', color: '#4338ca', fontWeight: 600 }}
                            />
                        </Box>
                    </Box>
                    <IconButton onClick={onClose} size="small" sx={{ color: '#94a3b8' }}>
                        <CloseIcon fontSize="small" />
                    </IconButton>
                </Box>
            </DialogTitle>

            <DialogContent sx={{ p: 0 }}>
                <Box sx={{ px: 2.5, py: 2, display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {/* Explanation */}
                    {fund.explanation && (
                        <Box>
                            <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b', mb: 0.75, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <LightbulbIcon sx={{ fontSize: 14 }} /> {t('recommendations.detail.reason')}
                            </Typography>
                            <Typography sx={{ color: '#334155', fontSize: '0.85rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                                {fund.explanation}
                            </Typography>
                        </Box>
                    )}

                    {/* Factor Data Grid */}
                    <Box sx={{ bgcolor: '#f8fafc', borderRadius: '8px', p: 1.5 }}>
                        <Typography sx={{ fontSize: '0.7rem', fontWeight: 600, color: '#64748b', mb: 1 }}>
                            {isShortTerm ? t('recommendations.v2.momentum_factors') : t('recommendations.v2.risk_factors')}
                        </Typography>
                        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1.5 }}>
                            {isShortTerm ? (
                                <>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.sharpe_20d')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.sharpe_20d || 0) >= 1.5 ? '#16a34a' : '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.sharpe_20d?.toFixed(2) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.table.return_1m')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.return_1m || 0) > 0 ? '#dc2626' : '#16a34a', fontSize: '0.85rem' }}>
                                            {factors.return_1m?.toFixed(2) || '-'}%
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.volatility')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.volatility_60d || 0) < 15 ? '#16a34a' : '#f97316', fontSize: '0.85rem' }}>
                                            {factors.volatility_60d?.toFixed(1) || '-'}%
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.table.return_1w')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.return_1w || 0) > 0 ? '#dc2626' : '#16a34a', fontSize: '0.85rem' }}>
                                            {factors.return_1w?.toFixed(2) || '-'}%
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.momentum_score')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.momentum_score?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                </>
                            ) : (
                                <>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.sharpe_1y')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.sharpe_1y || 0) >= 1.0 ? '#16a34a' : '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.sharpe_1y?.toFixed(2) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.sortino')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.sortino_1y?.toFixed(2) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.max_dd')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: Math.abs(factors.max_drawdown_1y || 0) < 15 ? '#16a34a' : '#dc2626', fontSize: '0.85rem' }}>
                                            {factors.max_drawdown_1y?.toFixed(1) || '-'}%
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.tenure')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.manager_tenure_years || 0) >= 3 ? '#16a34a' : '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.manager_tenure_years?.toFixed(1) || '-'}{t('common.years')}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.v2.alpha_score')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: '#1e293b', fontSize: '0.85rem' }}>
                                            {factors.alpha_score?.toFixed(0) || '-'}
                                        </Typography>
                                    </Box>
                                    <Box>
                                        <Typography sx={{ color: '#94a3b8', fontSize: '0.7rem' }}>{t('recommendations.table.return_1y')}</Typography>
                                        <Typography sx={{ fontWeight: 600, fontFamily: 'monospace', color: (factors.return_1y || 0) > 0 ? '#dc2626' : '#16a34a', fontSize: '0.85rem' }}>
                                            {factors.return_1y?.toFixed(2) || '-'}%
                                        </Typography>
                                    </Box>
                                </>
                            )}
                        </Box>
                    </Box>

                    {/* Catalysts & Risks */}
                    {((fund.catalysts && fund.catalysts.length > 0) || (fund.risks && fund.risks.length > 0)) && (
                        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
                            {fund.catalysts && fund.catalysts.length > 0 && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#16a34a', mb: 0.75 }}>{t('recommendations.detail.catalysts')}</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {fund.catalysts.map((catalyst, i) => (
                                            <Typography key={i} sx={{ color: '#475569', fontSize: '0.8rem', pl: 1.5, position: 'relative', '&::before': { content: '"•"', position: 'absolute', left: 0, color: '#16a34a' } }}>
                                                {catalyst}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                            {fund.risks && fund.risks.length > 0 && (
                                <Box>
                                    <Typography sx={{ fontSize: '0.75rem', fontWeight: 600, color: '#dc2626', mb: 0.75 }}>{t('recommendations.detail.risk_factors')}</Typography>
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                                        {fund.risks.map((risk, i) => (
                                            <Typography key={i} sx={{ color: '#475569', fontSize: '0.8rem', pl: 1.5, position: 'relative', '&::before': { content: '"•"', position: 'absolute', left: 0, color: '#dc2626' } }}>
                                                {risk}
                                            </Typography>
                                        ))}
                                    </Box>
                                </Box>
                            )}
                        </Box>
                    )}
                </Box>
            </DialogContent>

            <DialogActions sx={{ px: 2.5, py: 1.5, borderTop: '1px solid #e2e8f0' }}>
                <Typography sx={{ color: '#94a3b8', fontSize: '0.75rem', flex: 1 }}>
                    {t('recommendations.v2.strategy')}: {fund.strategy || (isShortTerm ? t('recommendations.v2.momentum') : t('recommendations.v2.alpha'))}
                </Typography>
                <Button onClick={onClose} size="small" sx={{ color: '#64748b', fontSize: '0.8rem' }}>
                    {t('recommendations.detail.close')}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

// --- Main Page Component ---

export default function RecommendationsPage() {
    const { t } = useTranslation();
    const [data, setData] = useState<RecommendationResult | null>(null);
    const [dataV2, setDataV2] = useState<RecommendationResultV2 | null>(null);
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [generatingProgress, setGeneratingProgress] = useState<string>('');
    const [mode, setMode] = useState<'all' | 'short' | 'long'>('all');
    const [preferencesOpen, setPreferencesOpen] = useState(false);
    const [hasPreferences, setHasPreferences] = useState(false);
    const [useV2Engine, setUseV2Engine] = useState(false);
    const [factorStatus, setFactorStatus] = useState<FactorStatus | null>(null);
    const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: 'success' | 'error' }>({
        open: false,
        message: '',
        severity: 'success'
    });

    const loadData = async () => {
        try {
            setLoading(true);
            const result = await fetchLatestRecommendations();
            if (result.available && result.data) {
                setData(result.data);
            }
        } catch (error) {
            console.error('Failed to load recommendations', error);
        } finally {
            setLoading(false);
        }
    };

    const checkPreferences = async () => {
        try {
            const result = await getUserPreferences();
            setHasPreferences(result.exists);
        } catch (error) {
            console.error('Failed to check preferences', error);
        }
    };

    const checkFactorStatus = async () => {
        try {
            const status = await getFactorStatusV2();
            setFactorStatus(status);
        } catch (error) {
            console.error('Failed to check factor status', error);
        }
    };

    const handleGenerate = async (forceRefresh: boolean = false) => {
        try {
            setGenerating(true);
            setGeneratingProgress(t('recommendations.messages.generate_started'));
            setSnackbar({ open: true, message: t('recommendations.messages.generate_started'), severity: 'success' });

            // Use V2 engine if enabled
            if (useV2Engine) {
                const result = await generateRecommendationsV2({
                    mode,
                    stock_limit: 20,
                    fund_limit: 20,
                    use_llm: true
                });
                setDataV2(result);
                setSnackbar({ open: true, message: t('recommendations.messages.generate_success'), severity: 'success' });
                setGenerating(false);
                setGeneratingProgress('');
                return;
            }

            // V1 engine
            const response = await generateRecommendations({ mode, force_refresh: forceRefresh });

            // If cached result returned immediately
            if (response.status === 'completed' && response.result) {
                setData(response.result);
                setSnackbar({ open: true, message: t('recommendations.messages.generate_success'), severity: 'success' });
                setGenerating(false);
                setGeneratingProgress('');
                return;
            }

            // If task started in background, poll for completion
            if (response.status === 'started' && response.task_id) {
                const result = await pollRecommendationTask(
                    response.task_id,
                    (status: TaskStatusResponse) => {
                        setGeneratingProgress(status.progress || '');
                    },
                    5000,  // Poll every 5 seconds
                    200    // Max 200 attempts (~10 minutes)
                );
                setData(result);
                setSnackbar({ open: true, message: t('recommendations.messages.generate_success'), severity: 'success' });
            }
        } catch (error) {
            console.error('Failed to generate recommendations', error);
            setSnackbar({ open: true, message: t('recommendations.messages.generate_error'), severity: 'error' });
        } finally {
            setGenerating(false);
            setGeneratingProgress('');
        }
    };

    useEffect(() => {
        loadData();
        checkPreferences();
        checkFactorStatus();
    }, []);

    // Extract data based on mode and engine version
    const shortTermData = data?.short_term;
    const longTermData = data?.long_term;

    // V1 data
    const shortStocks = shortTermData?.short_term_stocks || shortTermData?.stocks || [];
    const shortFunds = shortTermData?.short_term_funds || shortTermData?.funds || [];
    const longStocks = longTermData?.long_term_stocks || longTermData?.stocks || [];
    const longFunds = longTermData?.long_term_funds || longTermData?.funds || [];

    // V2 data
    const shortStocksV2 = dataV2?.short_term?.stocks || [];
    const shortFundsV2 = dataV2?.short_term?.funds || [];
    const longStocksV2 = dataV2?.long_term?.stocks || [];
    const longFundsV2 = dataV2?.long_term?.funds || [];

    // Check if we have any data to display
    const hasV2Data = dataV2 && (shortStocksV2.length > 0 || longStocksV2.length > 0);
    const hasV1Data = data && (shortStocks.length > 0 || longStocks.length > 0);

    // Determine which data to display based on engine and data availability
    const displayV2 = useV2Engine && hasV2Data;

    return (
        <Box className="flex flex-col gap-6 w-full h-full pb-10">
            {/* Header */}
            <Box className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <Box className="flex items-center gap-3">
                    <Box className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
                        <AutoAwesomeIcon className="text-white" />
                    </Box>
                    <Box>
                        <Typography variant="h5" className="font-extrabold text-slate-800 tracking-tight">
                            {t('recommendations.title')}
                        </Typography>
                        <Typography variant="body2" className="text-slate-500">
                            {t('recommendations.subtitle')}
                        </Typography>
                    </Box>
                </Box>

                <Box className="flex items-center gap-3">
                    {/* V2 Engine Toggle */}
                    <Tooltip title={
                        useV2Engine
                            ? t('recommendations.v2.engine_enabled_tip')
                            : t('recommendations.v2.engine_disabled_tip')
                    }>
                        <Box
                            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border cursor-pointer transition-all ${
                                useV2Engine
                                    ? 'bg-emerald-50 border-emerald-200 text-emerald-700'
                                    : 'bg-slate-50 border-slate-200 text-slate-500 hover:bg-slate-100'
                            }`}
                            onClick={() => setUseV2Engine(!useV2Engine)}
                        >
                            <ScienceIcon fontSize="small" />
                            <Typography variant="caption" className="font-semibold">
                                {useV2Engine ? 'V2' : 'V1'}
                            </Typography>
                            {factorStatus?.stock_factors?.count && factorStatus.stock_factors.count > 0 && useV2Engine && (
                                <Badge
                                    badgeContent={factorStatus.stock_factors.count}
                                    color="success"
                                    max={9999}
                                    sx={{ '& .MuiBadge-badge': { fontSize: '0.6rem', height: 14, minWidth: 14 } }}
                                />
                            )}
                        </Box>
                    </Tooltip>

                    {/* Preferences Button */}
                    <Tooltip title={hasPreferences ? t('recommendations.preferences.tooltip_configured') : t('recommendations.preferences.tooltip_not_configured')}>
                        <IconButton
                            size="small"
                            onClick={() => setPreferencesOpen(true)}
                            className={`border shadow-sm hover:bg-slate-50 ${
                                hasPreferences
                                    ? 'bg-purple-50 border-purple-200'
                                    : 'bg-white border-slate-200'
                            }`}
                        >
                            <Badge
                                color="success"
                                variant="dot"
                                invisible={!hasPreferences}
                            >
                                <TuneIcon fontSize="small" className={hasPreferences ? 'text-purple-600' : 'text-slate-500'} />
                            </Badge>
                        </IconButton>
                    </Tooltip>

                    {/* Mode Selector */}
                    <ToggleButtonGroup
                        value={mode}
                        exclusive
                        onChange={(_, v) => v && setMode(v)}
                        size="small"
                        className="bg-slate-100 rounded-lg"
                    >
                        <ToggleButton value="all" className="px-4 text-xs font-semibold">
                            {t('recommendations.mode.all')}
                        </ToggleButton>
                        <ToggleButton value="short" className="px-4 text-xs font-semibold">
                            {t('recommendations.mode.short')}
                        </ToggleButton>
                        <ToggleButton value="long" className="px-4 text-xs font-semibold">
                            {t('recommendations.mode.long')}
                        </ToggleButton>
                    </ToggleButtonGroup>

                    {/* Generate Button */}
                    <Button
                        variant="contained"
                        startIcon={generating ? <CircularProgress size={16} color="inherit" /> : <AutoAwesomeIcon />}
                        onClick={() => handleGenerate(false)}
                        disabled={generating}
                        className="bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-md hover:shadow-lg"
                    >
                        {generating ? (generatingProgress || t('recommendations.generating')) : t('recommendations.generate')}
                    </Button>

                    {/* Force Refresh */}
                    <Tooltip title={t('recommendations.force_refresh')}>
                        <IconButton
                            size="small"
                            onClick={() => handleGenerate(true)}
                            disabled={generating}
                            className="bg-white border border-slate-200 shadow-sm hover:bg-slate-50"
                        >
                            <RefreshIcon fontSize="small" />
                        </IconButton>
                    </Tooltip>
                </Box>
            </Box>

            {/* Last Updated Info */}
            {(data?.generated_at || dataV2?.generated_at) && (
                <Box className="flex items-center gap-2 text-slate-500 flex-wrap">
                    <AccessTimeIcon className="text-sm" />
                    <Typography variant="caption">
                        {t('recommendations.last_updated')}: {new Date((displayV2 ? dataV2?.generated_at : data?.generated_at) || '').toLocaleString()}
                    </Typography>
                    {displayV2 && dataV2?.trade_date && (
                        <Chip
                            label={`${t('recommendations.v2.trade_date')}: ${dataV2.trade_date}`}
                            size="small"
                            className="h-5 text-[10px] bg-emerald-50 text-emerald-700 ml-2"
                        />
                    )}
                    {displayV2 && dataV2?.metadata && (
                        <Box className="flex gap-3 ml-4">
                            <Chip
                                label={`${t('recommendations.v2.factor_time')}: ${dataV2.metadata.factor_computation_time?.toFixed(1)}s`}
                                size="small"
                                className="h-5 text-[10px] bg-slate-100"
                            />
                            <Chip
                                label={`${t('recommendations.metadata.total_time')}: ${dataV2.metadata.total_time?.toFixed(1)}s`}
                                size="small"
                                className="h-5 text-[10px] bg-emerald-100 text-emerald-700"
                            />
                        </Box>
                    )}
                    {!displayV2 && data?.metadata && (
                        <Box className="flex gap-3 ml-4">
                            <Chip
                                label={`${t('recommendations.metadata.screening_time')}: ${data.metadata.screening_time?.toFixed(1)}s`}
                                size="small"
                                className="h-5 text-[10px] bg-slate-100"
                            />
                            <Chip
                                label={`${t('recommendations.metadata.llm_time')}: ${data.metadata.llm_time?.toFixed(1)}s`}
                                size="small"
                                className="h-5 text-[10px] bg-slate-100"
                            />
                            <Chip
                                label={`${t('recommendations.metadata.total_time')}: ${data.metadata.total_time?.toFixed(1)}s`}
                                size="small"
                                className="h-5 text-[10px] bg-blue-100 text-blue-700"
                            />
                        </Box>
                    )}
                    {hasPreferences && !displayV2 && (
                        <Chip
                            icon={<TuneIcon className="text-[12px]" />}
                            label={data?.personalized ? t('recommendations.preferences.personalized') : t('recommendations.preferences.preferences_set')}
                            size="small"
                            className={`h-5 text-[10px] ml-2 ${
                                data?.personalized
                                    ? 'bg-purple-100 text-purple-700'
                                    : 'bg-slate-100 text-slate-600'
                            }`}
                        />
                    )}
                    {displayV2 && (
                        <Chip
                            icon={<ScienceIcon sx={{ fontSize: 12 }} />}
                            label={t('recommendations.v2.quant_engine')}
                            size="small"
                            className="h-5 text-[10px] ml-2 bg-emerald-100 text-emerald-700"
                        />
                    )}
                </Box>
            )}

            {/* Loading State */}
            {loading && (
                <Box className="flex items-center justify-center py-20">
                    <CircularProgress size={32} className="text-slate-400" />
                </Box>
            )}

            {/* No Data State */}
            {!loading && !data && !dataV2 && (
                <Paper elevation={0} className="border border-slate-200 rounded-xl bg-white p-12 text-center">
                    <AutoAwesomeIcon className="text-6xl text-slate-300 mb-4" />
                    <Typography variant="h6" className="text-slate-600 mb-2">
                        {t('recommendations.no_data')}
                    </Typography>
                    <Typography variant="body2" className="text-slate-400 mb-6">
                        {t('recommendations.no_data_hint')}
                    </Typography>
                    <Button
                        variant="contained"
                        startIcon={<AutoAwesomeIcon />}
                        onClick={() => handleGenerate(false)}
                        disabled={generating}
                        className="bg-gradient-to-r from-blue-600 to-purple-600"
                    >
                        {t('recommendations.generate')}
                    </Button>
                </Paper>
            )}

            {/* V2 Recommendations Content */}
            {!loading && displayV2 && dataV2 && (
                <Box className="flex flex-col gap-6">
                    {/* V2 Short-Term Section */}
                    {(mode === 'all' || mode === 'short') && dataV2.short_term && (
                        <RecommendationSectionV2
                            title={t('recommendations.short_term.title')}
                            subtitle={t('recommendations.v2.short_term_subtitle')}
                            icon={<TrendingUpIcon className="text-blue-600" />}
                            stocks={shortStocksV2}
                            funds={shortFundsV2}
                            marketView={dataV2.short_term.market_view}
                            isShortTerm={true}
                            defaultExpanded={true}
                        />
                    )}

                    {/* V2 Long-Term Section */}
                    {(mode === 'all' || mode === 'long') && dataV2.long_term && (
                        <RecommendationSectionV2
                            title={t('recommendations.long_term.title')}
                            subtitle={t('recommendations.v2.long_term_subtitle')}
                            icon={<CalendarMonthIcon className="text-purple-600" />}
                            stocks={longStocksV2}
                            funds={longFundsV2}
                            marketView={dataV2.long_term.macro_view}
                            isShortTerm={false}
                            defaultExpanded={mode !== 'all'}
                        />
                    )}
                </Box>
            )}

            {/* V1 Recommendations Content */}
            {!loading && !displayV2 && data && (
                <Box className="flex flex-col gap-6">
                    {/* Short-Term Section */}
                    {(mode === 'all' || mode === 'short') && shortTermData && (
                        <RecommendationSection
                            title={t('recommendations.short_term.title')}
                            subtitle={t('recommendations.short_term.subtitle')}
                            icon={<TrendingUpIcon className="text-blue-600" />}
                            stocks={shortStocks}
                            funds={shortFunds}
                            marketView={shortTermData.market_view}
                            sectorPreference={shortTermData.sector_preference}
                            riskWarning={shortTermData.risk_warning}
                            isShortTerm={true}
                            defaultExpanded={true}
                        />
                    )}

                    {/* Long-Term Section */}
                    {(mode === 'all' || mode === 'long') && longTermData && (
                        <RecommendationSection
                            title={t('recommendations.long_term.title')}
                            subtitle={t('recommendations.long_term.subtitle')}
                            icon={<CalendarMonthIcon className="text-purple-600" />}
                            stocks={longStocks}
                            funds={longFunds}
                            marketView={longTermData.macro_view}
                            sectorPreference={longTermData.sector_preference}
                            riskWarning={longTermData.risk_warning}
                            isShortTerm={false}
                            defaultExpanded={mode !== 'all'}
                        />
                    )}
                </Box>
            )}

            {/* Snackbar */}
            <Snackbar
                open={snackbar.open}
                autoHideDuration={4000}
                onClose={() => setSnackbar({ ...snackbar, open: false })}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
            >
                <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })}>
                    {snackbar.message}
                </Alert>
            </Snackbar>

            {/* Preferences Modal */}
            <PreferencesModal
                open={preferencesOpen}
                onClose={() => setPreferencesOpen(false)}
                onSaved={() => {
                    setHasPreferences(true);
                    setSnackbar({
                        open: true,
                        message: t('recommendations.preferences.save_success_hint'),
                        severity: 'success'
                    });
                }}
            />
        </Box>
    );
}
