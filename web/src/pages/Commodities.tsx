import { useState, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
    Box,
    Typography,
    Button,
    CircularProgress,
    IconButton,
    TextField,
    Tooltip,
    Paper,
    Chip,
    InputAdornment
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import DownloadIcon from '@mui/icons-material/Download';
import SearchIcon from '@mui/icons-material/Search';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import html2canvas from 'html2canvas';
import { fetchCommodityReports, fetchReportContent, generateCommodityReport, deleteCommodityReport } from '../api';
import type { ReportSummary } from '../api';

export default function CommoditiesPage() {
    const { t } = useTranslation();
    const [reports, setReports] = useState<ReportSummary[]>([]);
    const [selectedReport, setSelectedReport] = useState<ReportSummary | null>(null);
    const [reportContent, setReportContent] = useState<string>('');
    const [loadingContent, setLoadingContent] = useState<boolean>(false);
    const [generatingGold, setGeneratingGold] = useState<boolean>(false);
    const [generatingSilver, setGeneratingSilver] = useState<boolean>(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [exporting, setExporting] = useState<boolean>(false);

    const reportRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        loadReports();
    }, []);

    const loadReports = async () => {
        try {
            const commodityReports = await fetchCommodityReports();
            setReports(commodityReports);
            // Auto-select latest if available and none currently selected
            if (commodityReports.length > 0 && !selectedReport) {
                handleSelectReport(commodityReports[0]);
            }
        } catch (error) {
            console.error("Failed to load reports", error);
        }
    };

    const handleSelectReport = async (report: ReportSummary) => {
        setSelectedReport(report);
        setLoadingContent(true);
        try {
            const content = await fetchReportContent(report.filename);
            setReportContent(content);
        } catch (error) {
            console.error("Failed to load content", error);
        } finally {
            setLoadingContent(false);
        }
    };

    const handleDeleteReport = async (e: React.MouseEvent, filename: string) => {
        e.stopPropagation();
        if (window.confirm('Are you sure you want to delete this report?')) {
            try {
                await deleteCommodityReport(filename);
                // Update local state immediately
                const updatedReports = reports.filter(r => r.filename !== filename);
                setReports(updatedReports);
                
                if (selectedReport?.filename === filename) {
                    setSelectedReport(null);
                    setReportContent('');
                    
                    // Optional: Select next available if any
                    if (updatedReports.length > 0) {
                        handleSelectReport(updatedReports[0]);
                    }
                }
            } catch (error) {
                console.error("Failed to delete report", error);
                alert("Failed to delete report");
            }
        }
    };

    const handleGenerate = async (asset: 'gold' | 'silver') => {
        if (asset === 'gold') setGeneratingGold(true);
        else setGeneratingSilver(true);
        
        try {
            await generateCommodityReport(asset);
            await loadReports();
        } catch (error) {
            console.error("Analysis failed", error);
        } finally {
            if (asset === 'gold') setGeneratingGold(false);
            else setGeneratingSilver(false);
        }
    };

    const handleExportImage = async () => {
        if (!reportRef.current || !selectedReport) return;
        setExporting(true);
        try {
            const canvas = await html2canvas(reportRef.current, { scale: 2 });
            const link = document.createElement('a');
            link.download = `Commodity_${selectedReport.fund_name}_${selectedReport.date}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
        } finally {
            setExporting(false);
        }
    };

    // Grouping Logic
    const groupedReports = useMemo(() => {
        const filtered = reports.filter(r => 
            r.fund_name?.toLowerCase().includes(searchQuery.toLowerCase()) || 
            r.date.includes(searchQuery)
        );
        
        const groups: Record<string, ReportSummary[]> = {};
        filtered.forEach(r => {
            // date format from API is "YYYY-MM-DD HH:MM:SS" or just date. 
            // We want to group by just YYYY-MM-DD
            const day = r.date.split(' ')[0];
            if (!groups[day]) groups[day] = [];
            groups[day].push(r);
        });
        
        return Object.keys(groups).sort((a, b) => b.localeCompare(a)).map(date => ({
            date,
            items: groups[date].sort((a, _b) => (a.fund_code === 'gold' ? -1 : 1))
        }));
    }, [reports, searchQuery]);

    return (
        <div className="h-full flex flex-col bg-slate-50 overflow-hidden">
            {/* Asset Dashboard Header - Ultra Minimalist */}
            <Box sx={{ px: 4, py: 3, bgcolor: '#fff', borderBottom: '1px solid #f1f5f9', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                    <Typography variant="h6" sx={{ fontWeight: 800, color: '#0f172a', fontFamily: 'JetBrains Mono', letterSpacing: '-0.02em' }}>
                        {t('commodities.title_main')}
                    </Typography>
                    <Typography variant="caption" sx={{ color: '#94a3b8', fontWeight: 600 }}>
                        {t('commodities.subtitle')}
                    </Typography>
                </div>
                
                <Box sx={{ display: 'flex', gap: 2 }}>
                    <Button
                        variant="outlined"
                        size="small"
                        onClick={() => handleGenerate('gold')}
                        disabled={generatingGold}
                        startIcon={generatingGold ? <CircularProgress size={14} color="inherit" /> : <AutoAwesomeIcon sx={{ fontSize: 16 }} />}
                        sx={{ 
                            color: '#d97706', 
                            borderColor: '#fcd34d',
                            bgcolor: 'rgba(251, 191, 36, 0.05)',
                            '&:hover': { bgcolor: 'rgba(251, 191, 36, 0.1)', borderColor: '#d97706' },
                            fontWeight: 700,
                            borderRadius: '20px',
                            textTransform: 'none',
                            fontSize: '0.75rem',
                            px: 2
                        }}
                    >
                        {generatingGold ? t('commodities.analyzing_gold') : t('commodities.analyze_gold')}
                    </Button>

                    <Button
                        variant="outlined"
                        size="small"
                        onClick={() => handleGenerate('silver')}
                        disabled={generatingSilver}
                        startIcon={generatingSilver ? <CircularProgress size={14} color="inherit" /> : <AutoAwesomeIcon sx={{ fontSize: 16 }} />}
                        sx={{ 
                            color: '#475569', 
                            borderColor: '#cbd5e1',
                            bgcolor: 'rgba(241, 245, 249, 0.5)',
                            '&:hover': { bgcolor: '#f1f5f9', borderColor: '#94a3b8' },
                            fontWeight: 700,
                            borderRadius: '20px',
                            textTransform: 'none',
                            fontSize: '0.75rem',
                            px: 2
                        }}
                    >
                        {generatingSilver ? t('commodities.analyzing_silver') : t('commodities.analyze_silver')}
                    </Button>
                </Box>
            </Box>

            {/* Split View Content */}
            <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                {/* Left: Report Timeline */}
                <Box sx={{ width: 320, borderRight: '1px solid #f1f5f9', bgcolor: '#fff', display: 'flex', flexDirection: 'column' }}>
                    <Box sx={{ p: 2, borderBottom: '1px solid #f1f5f9' }}>
                        <TextField
                            fullWidth
                            size="small"
                            placeholder={t('commodities.filter_timeline')}
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            InputProps={{
                                startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" className="text-slate-400"/></InputAdornment>,
                                endAdornment: <IconButton size="small" onClick={loadReports}><RefreshIcon fontSize="small" /></IconButton>
                            }}
                            sx={{ '& .MuiOutlinedInput-root': { bgcolor: '#f8fafc', borderRadius: '8px' } }}
                        />
                    </Box>
                    <Box sx={{ flex: 1, overflowY: 'auto', p: 2 }} className="custom-scrollbar">
                        {groupedReports.map((group) => (
                            <Box key={group.date} sx={{ mb: 3 }}>
                                <Typography sx={{ 
                                    fontSize: '0.75rem', 
                                    fontWeight: 800, 
                                    color: '#94a3b8', 
                                    mb: 1.5, 
                                    pl: 1,
                                    fontFamily: 'JetBrains Mono'
                                }}>
                                    {group.date}
                                </Typography>
                                {group.items.map(report => {
                                    const isGold = report.fund_code === 'gold';
                                    const isSelected = selectedReport?.filename === report.filename;
                                    return (
                                        <Box 
                                            key={report.filename}
                                            onClick={() => handleSelectReport(report)}
                                            sx={{
                                                p: 2,
                                                pr: 5, // Reserve space for delete button
                                                mb: 1,
                                                borderRadius: '12px',
                                                cursor: 'pointer',
                                                border: isSelected ? (isGold ? '1px solid #fcd34d' : '1px solid #cbd5e1') : '1px solid transparent',
                                                bgcolor: isSelected ? (isGold ? '#fffbeb' : '#f8fafc') : 'transparent',
                                                '&:hover': { bgcolor: isGold ? '#fffbeb' : '#f8fafc' },
                                                transition: 'all 0.2s',
                                                position: 'relative',
                                                '&:hover .delete-btn': { opacity: 1 }
                                            }}
                                        >
                                            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                                                <Typography sx={{ fontWeight: 700, fontSize: '0.9rem', color: isGold ? '#b45309' : '#334155' }}>
                                                    {isGold ? t('commodities.gold_strategy') : t('commodities.silver_strategy')}
                                                </Typography>
                                                {isGold ? <ShowChartIcon sx={{ fontSize: 16, color: '#d97706' }} /> : <TrendingUpIcon sx={{ fontSize: 16, color: '#64748b' }} />}
                                            </Box>
                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                <AccessTimeIcon sx={{ fontSize: 12, color: '#94a3b8' }} />
                                                <Typography sx={{ fontSize: '0.7rem', color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>
                                                    {t('commodities.intel_report')}
                                                </Typography>
                                            </Box>
                                            
                                            <IconButton
                                                className="delete-btn"
                                                size="small"
                                                onClick={(e) => handleDeleteReport(e, report.filename)}
                                                sx={{
                                                    position: 'absolute',
                                                    right: 4,
                                                    top: '50%',
                                                    transform: 'translateY(-50%)',
                                                    opacity: 0,
                                                    color: '#94a3b8',
                                                    transition: 'all 0.2s',
                                                    padding: '4px',
                                                    '&:hover': { color: '#ef4444', bgcolor: 'rgba(239,68,68,0.1)' }
                                                }}
                                            >
                                                <DeleteOutlineIcon sx={{ fontSize: 18 }} />
                                            </IconButton>
                                        </Box>
                                    )
                                })}
                            </Box>
                        ))}
                    </Box>
                </Box>

                {/* Right: Reading Pane */}
                <Box sx={{ flex: 1, overflowY: 'auto', p: 4, bgcolor: '#f8fafc' }}>
                    {selectedReport ? (
                        <Box sx={{ maxWidth: '900px', mx: 'auto' }}>
                            <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
                                <Tooltip title={t('reports.download_image')}>
                                    <Button
                                        variant="outlined"
                                        size="small"
                                        onClick={handleExportImage}
                                        disabled={exporting || loadingContent}
                                        startIcon={exporting ? <CircularProgress size={14} /> : <DownloadIcon />}
                                        sx={{ color: '#64748b', borderColor: '#cbd5e1', bgcolor: '#fff', '&:hover': { bgcolor: '#f1f5f9' } }}
                                    >
                                        {t('reports.export')}
                                    </Button>
                                </Tooltip>
                            </Box>

                            <Paper 
                                ref={reportRef}
                                elevation={0} 
                                sx={{ 
                                    borderRadius: '16px', 
                                    overflow: 'hidden', 
                                    border: '1px solid #e2e8f0',
                                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.05)' 
                                }}
                            >
                                {/* Report Header */}
                                <Box sx={{ 
                                    p: 5, 
                                    background: selectedReport.fund_code === 'gold' 
                                        ? 'linear-gradient(to right, #fffbeb, #fff)' 
                                        : 'linear-gradient(to right, #f8fafc, #fff)',
                                    borderBottom: '1px solid #f1f5f9'
                                }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                        <div>
                                            <Typography variant="overline" sx={{ color: selectedReport.fund_code === 'gold' ? '#d97706' : '#64748b', fontWeight: 800, letterSpacing: '0.1em' }}>
                                                {t('commodities.brief')}
                                            </Typography>
                                            <Typography variant="h3" sx={{ fontWeight: 800, color: '#0f172a', mt: 1, mb: 2 }}>
                                                {selectedReport.fund_name}
                                            </Typography>
                                            <Box sx={{ display: 'flex', gap: 1.5 }}>
                                                <Chip 
                                                    label={selectedReport.date} 
                                                    size="small" 
                                                    icon={<CalendarTodayIcon style={{fontSize: 14}} />}
                                                    sx={{ bgcolor: '#fff', border: '1px solid #e2e8f0', fontWeight: 600, color: '#64748b' }} 
                                                />
                                                <Chip 
                                                    label={selectedReport.fund_code === 'gold' ? "XAU/USD" : "XAG/USD"} 
                                                    size="small" 
                                                    variant="outlined"
                                                    sx={{ borderColor: '#cbd5e1', color: '#64748b', fontFamily: 'JetBrains Mono', fontWeight: 600 }} 
                                                />
                                            </Box>
                                        </div>
                                        <Box sx={{ 
                                            width: 64, 
                                            height: 64, 
                                            borderRadius: '16px', 
                                            bgcolor: selectedReport.fund_code === 'gold' ? '#fcd34d' : '#cbd5e1',
                                            display: 'flex', 
                                            alignItems: 'center', 
                                            justifyContent: 'center',
                                            fontSize: '1.5rem',
                                            fontWeight: 900,
                                            color: '#fff',
                                            boxShadow: '0 10px 15px -3px rgba(0,0,0,0.1)'
                                        }}>
                                            {selectedReport.fund_code === 'gold' ? 'Au' : 'Ag'}
                                        </Box>
                                    </Box>
                                </Box>

                                {/* Report Body */}
                                <Box sx={{ p: 6, bgcolor: '#fff', minHeight: '600px' }} className="markdown-body">
                                    {loadingContent ? (
                                        <Box sx={{ display: 'flex', justifyContent: 'center', py: 20 }}>
                                            <CircularProgress sx={{ color: selectedReport.fund_code === 'gold' ? '#d97706' : '#64748b' }} />
                                        </Box>
                                    ) : (
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                            {reportContent}
                                        </ReactMarkdown>
                                    )}
                                </Box>

                                {/* Footer */}
                                <Box sx={{ p: 3, bgcolor: '#f8fafc', borderTop: '1px solid #f1f5f9', textAlign: 'center' }}>
                                    <Typography variant="caption" sx={{ color: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: '0.65rem', fontWeight: 700, letterSpacing: '0.1em' }}>
                                        {t('commodities.footer')} • {new Date().getFullYear()}
                                    </Typography>
                                </Box>
                            </Paper>
                        </Box>
                    ) : (
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#94a3b8' }}>
                            <TrendingUpIcon sx={{ fontSize: 64, mb: 2, opacity: 0.2 }} />
                            <Typography sx={{ fontWeight: 600 }}>{t('commodities.select_prompt')}</Typography>
                        </Box>
                    )}
                </Box>
            </Box>
        </div>
    );
}