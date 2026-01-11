import { useState, useEffect, useMemo, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
    Box,
    Typography,
    List,
    ListItemButton,
    ListItemText,
    Chip,
    Button,
    CircularProgress,
    IconButton,
    Collapse,
    ListItemIcon,
    TextField,
    InputAdornment,
    Tooltip
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import DownloadIcon from '@mui/icons-material/Download';
import DeleteIcon from '@mui/icons-material/Delete';
import html2canvas from 'html2canvas';
import ExpandLess from '@mui/icons-material/ExpandLess';
import ExpandMore from '@mui/icons-material/ExpandMore';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';
import ArticleIcon from '@mui/icons-material/Article';
import BusinessCenterIcon from '@mui/icons-material/BusinessCenter';
import SearchIcon from '@mui/icons-material/Search';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { fetchReports, fetchReportContent, deleteReport } from '../api';
import type { ReportSummary } from '../api'

export default function ReportsPage() {
    const { t } = useTranslation();
    const [reports, setReports] = useState<ReportSummary[]>([]);
    const [selectedReport, setSelectedReport] = useState<ReportSummary | null>(null);
    const [reportContent, setReportContent] = useState<string>('');
    const [loadingContent, setLoadingContent] = useState<boolean>(false);
    const [exporting, setExporting] = useState<boolean>(false);

    // Ref for export
    const reportRef = useRef<HTMLDivElement>(null);

    // Search State
    const [searchQuery, setSearchQuery] = useState('');

    // Grouping State
    const [expandedDates, setExpandedDates] = useState<Set<string>>(new Set());
    const [expandedFunds, setExpandedFunds] = useState<Set<string>>(new Set());

    useEffect(() => {
        loadReports();
    }, []);

    const loadReports = async () => {
        try {
            const data = await fetchReports();
            setReports(data);

            // Auto-expand the first date if not already expanded (optional, user might want to keep state)
            if (data.length > 0 && expandedDates.size === 0) {
                const newestDate = data[0].date;
                setExpandedDates(new Set([newestDate]));

                if (!selectedReport) {
                    handleSelectReport(data[0]);
                }
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
        e.stopPropagation(); // Prevent selecting the report when clicking delete
        if (!window.confirm('Are you sure you want to delete this report?')) return;

        try {
            await deleteReport(filename);
            
            // If the deleted report was selected, clear selection
            if (selectedReport?.filename === filename) {
                setSelectedReport(null);
                setReportContent('');
            }
            
            // Reload list
            await loadReports();
        } catch (error) {
            console.error("Failed to delete report", error);
            alert("Failed to delete report");
        }
    };

    const toggleDate = (date: string) => {
        const newSet = new Set(expandedDates);
        if (newSet.has(date)) newSet.delete(date);
        else newSet.add(date);
        setExpandedDates(newSet);
    };

    const toggleFund = (fundKey: string) => {
        const newSet = new Set(expandedFunds);
        if (newSet.has(fundKey)) newSet.delete(fundKey);
        else newSet.add(fundKey);
        setExpandedFunds(newSet);
    };

    // Export report as image
    const handleExportImage = async () => {
        if (!reportRef.current || !selectedReport) return;
        
        setExporting(true);
        try {
            // 设置固定宽度以获得更好的导出效果
            const exportWidth = 1200; // 固定导出宽度，接近页面显示宽度
            const originalWidth = reportRef.current.style.width;
            const originalMaxWidth = reportRef.current.style.maxWidth;
            
            // 临时设置宽度
            reportRef.current.style.width = `${exportWidth}px`;
            reportRef.current.style.maxWidth = `${exportWidth}px`;
            
            const canvas = await html2canvas(reportRef.current, {
                scale: 2, // Higher quality
                useCORS: true,
                backgroundColor: '#18181b', // Matches bg-surface
                logging: false,
                width: exportWidth,
                windowWidth: exportWidth
            });
            
            // 恢复原始样式
            reportRef.current.style.width = originalWidth;
            reportRef.current.style.maxWidth = originalMaxWidth;
            
            // Convert to image and download
            const link = document.createElement('a');
            const fileName = `${selectedReport.fund_name || 'report'}_${selectedReport.mode}_${selectedReport.date}.png`;
            link.download = fileName.replace(/[^a-zA-Z0-9_\-一-龥]/g, '_');
            link.href = canvas.toDataURL('image/png');
            link.click();
        } catch (error) {
            console.error('Failed to export image:', error);
        } finally {
            setExporting(false);
        }
    };

    // Advanced Grouping Logic: Search -> Date -> Fund -> Report
    const groupedData = useMemo(() => {
        // 1. Filter
        const filtered = reports.filter(r => {
            const q = searchQuery.toLowerCase();
            return (
                r.date.includes(q) ||
                (r.fund_name && r.fund_name.toLowerCase().includes(q)) ||
                (r.fund_code && r.fund_code.includes(q))
            );
        });

        // 2. Group by Date
        const dateGroups: Record<string, { overview: ReportSummary[], funds: Record<string, ReportSummary[]> }> = {};

        filtered.forEach(r => {
            if (!dateGroups[r.date]) {
                dateGroups[r.date] = { overview: [], funds: {} };
            }

            if (r.is_summary) {
                dateGroups[r.date].overview.push(r);
            } else {
                const fundKey = `${r.fund_name}|${r.fund_code}`;
                if (!dateGroups[r.date].funds[fundKey]) {
                    dateGroups[r.date].funds[fundKey] = [];
                }
                dateGroups[r.date].funds[fundKey].push(r);
            }
        });

        // 3. Convert to Array and Sort
        return Object.keys(dateGroups).sort((a, b) => b.localeCompare(a)).map(date => {
            const group = dateGroups[date];
            const fundList = Object.keys(group.funds).map(key => {
                const [name, code] = key.split('|');
                return {
                    key,
                    name,
                    code,
                    reports: group.funds[key].sort((a, _b) => (a.mode === 'pre' ? -1 : 1)) // Pre first
                };
            });

            return {
                date,
                overviews: group.overview,
                funds: fundList
            };
        });
    }, [reports, searchQuery]);

    return (
        <div className="flex h-screen bg-background overflow-hidden text-slate-700">
            {/* Sidebar */}
            <div className="w-[360px] border-r border-slate-200 bg-white flex flex-col flex-shrink-0 z-10">
                {/* Header & Search */}
                <div className="p-4 border-b border-slate-200">
                    <div className="flex justify-between items-center mb-4">
                        <Typography variant="h6" className="font-bold text-slate-900 tracking-tight">{t('reports.library')}</Typography>
                        <IconButton size="small" onClick={loadReports} className="text-slate-400 hover:text-slate-700"><RefreshIcon /></IconButton>
                    </div>
                    <TextField
                        fullWidth
                        size="small"
                        placeholder={t('reports.search_placeholder')}
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        InputProps={{
                            startAdornment: <InputAdornment position="start"><SearchIcon fontSize="small" className="text-slate-400" /></InputAdornment>,
                        }}
                        sx={{ 
                            '& .MuiOutlinedInput-root': { bgcolor: '#f8fafc' },
                            '& input': { color: '#0f172a' }
                        }}
                    />
                </div>

                {/* Hierarchical List */}
                <div className="overflow-y-auto flex-1 custom-scrollbar">
                    <List component="nav" disablePadding>
                        {groupedData.map((group) => (
                            <Box key={group.date}>
                                {/* Level 1: Date */}
                                <ListItemButton onClick={() => toggleDate(group.date)} className="bg-slate-50 py-3 border-b border-slate-100 hover:bg-slate-100">
                                    <ListItemIcon sx={{ minWidth: 32 }}>
                                        <CalendarTodayIcon fontSize="small" sx={{ fontSize: 16, color: '#64748b' }} />
                                    </ListItemIcon>
                                    <ListItemText
                                        primary={group.date}
                                        primaryTypographyProps={{ fontWeight: 600, fontSize: '0.8rem', color: '#334155', fontFamily: 'monospace' }}
                                    />
                                    {expandedDates.has(group.date) ? <ExpandLess fontSize="small" sx={{ color: '#94a3b8' }} /> : <ExpandMore fontSize="small" sx={{ color: '#94a3b8' }} />}
                                </ListItemButton>

                                <Collapse in={expandedDates.has(group.date)} timeout="auto" unmountOnExit>
                                    <List component="div" disablePadding>

                                        {/* Level 2: Overviews (Directly listed) */}
                                        {group.overviews.map(report => (
                                            <ListItemButton
                                                key={report.filename}
                                                sx={{ 
                                                    pl: 5, 
                                                    borderLeft: selectedReport?.filename === report.filename ? '3px solid #2563eb' : '3px solid transparent',
                                                    bgcolor: selectedReport?.filename === report.filename ? '#eff6ff' : 'transparent',
                                                    '&:hover .delete-icon': { opacity: 1 }
                                                }}
                                                selected={selectedReport?.filename === report.filename}
                                                onClick={() => handleSelectReport(report)}
                                                className="hover:bg-slate-50 group"
                                            >
                                                <ListItemIcon sx={{ minWidth: 28 }}>
                                                    <ArticleIcon fontSize="small" sx={{ fontSize: 18, color: report.mode === 'pre' ? '#2563eb' : '#d97706' }} />
                                                </ListItemIcon>
                                                <ListItemText
                                                    primary={report.mode === 'pre' ? t('reports.daily_briefing') : t('reports.market_wrap')}
                                                    secondary={report.mode === 'pre' ? 'PRE-MARKET' : 'POST-MARKET'}
                                                    primaryTypographyProps={{ fontSize: '0.85rem', fontWeight: 500, color: '#0f172a' }}
                                                    secondaryTypographyProps={{ fontSize: '0.65rem', color: '#64748b', letterSpacing: '0.05em' }}
                                                />
                                                <IconButton 
                                                    size="small" 
                                                    onClick={(e) => handleDeleteReport(e, report.filename)}
                                                    className="delete-icon transition-opacity opacity-0 hover:text-red-500 text-slate-300"
                                                >
                                                    <DeleteIcon fontSize="small" style={{ fontSize: 16 }} />
                                                </IconButton>
                                            </ListItemButton>
                                        ))}

                                        {/* Level 2: Funds Group */}
                                        {group.funds.map(fund => {
                                            const isFundExpanded = expandedFunds.has(`${group.date}-${fund.key}`);
                                            return (
                                                <Box key={fund.key}>
                                                    <ListItemButton
                                                        onClick={() => toggleFund(`${group.date}-${fund.key}`)}
                                                        sx={{ pl: 4, py: 1 }}
                                                        className="hover:bg-slate-50"
                                                    >
                                                        <ListItemIcon sx={{ minWidth: 28 }}>
                                                            <BusinessCenterIcon fontSize="small" sx={{ fontSize: 18, color: '#94a3b8' }} />
                                                        </ListItemIcon>
                                                        <ListItemText
                                                            primary={fund.name}
                                                            secondary={fund.code}
                                                            primaryTypographyProps={{ fontSize: '0.85rem', fontWeight: 500, color: '#334155' }}
                                                            secondaryTypographyProps={{ fontSize: '0.7rem', fontFamily: 'monospace', color: '#64748b' }}
                                                        />
                                                        {isFundExpanded ? <ExpandLess fontSize="small" sx={{ fontSize: 14, color: '#cbd5e1' }} /> : <ExpandMore fontSize="small" sx={{ fontSize: 14, color: '#cbd5e1' }} />}
                                                    </ListItemButton>

                                                    {/* Level 3: Fund Reports */}
                                                    <Collapse in={isFundExpanded} timeout="auto" unmountOnExit>
                                                        <List component="div" disablePadding className="bg-slate-50/50">
                                                            {fund.reports.map(report => (
                                                                <ListItemButton
                                                                    key={report.filename}
                                                                    sx={{
                                                                        pl: 8,
                                                                        py: 1,
                                                                        borderLeft: selectedReport?.filename === report.filename ? '3px solid #2563eb' : '3px solid transparent',
                                                                        bgcolor: selectedReport?.filename === report.filename ? '#eff6ff' : 'transparent',
                                                                        '&:hover .delete-icon': { opacity: 1 }
                                                                    }}
                                                                    selected={selectedReport?.filename === report.filename}
                                                                    onClick={() => handleSelectReport(report)}
                                                                    className="hover:bg-slate-100 group"
                                                                >
                                                                    <ListItemIcon sx={{ minWidth: 24 }}>
                                                                        {report.mode === 'pre' ?
                                                                            <TrendingUpIcon sx={{ fontSize: 16, color: '#2563eb' }} /> :
                                                                            <TrendingDownIcon sx={{ fontSize: 16, color: '#d97706' }} />
                                                                        }
                                                                    </ListItemIcon>
                                                                    <ListItemText
                                                                        primary={report.mode === 'pre' ? t('reports.strategy_analysis') : t('reports.performance_review')}
                                                                        primaryTypographyProps={{ fontSize: '0.8rem', color: '#334155' }}
                                                                    />
                                                                    <IconButton 
                                                                        size="small" 
                                                                        onClick={(e) => handleDeleteReport(e, report.filename)}
                                                                        className="delete-icon transition-opacity opacity-0 hover:text-red-500 text-slate-300"
                                                                    >
                                                                        <DeleteIcon fontSize="small" style={{ fontSize: 16 }} />
                                                                    </IconButton>
                                                                </ListItemButton>
                                                            ))}
                                                        </List>
                                                    </Collapse>
                                                </Box>
                                            );
                                        })}
                                    </List>
                                </Collapse>
                            </Box>
                        ))}
                    </List>
                </div>
            </div>

            {/* Right Content: Professional Report View */}
            <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-50">
                {/* Export Button Bar */}
                {selectedReport && (
                    <div className="p-3 border-b border-slate-200 bg-white/80 flex justify-end gap-2 backdrop-blur-sm">
                        <Tooltip title={t('reports.download_image')}>
                            <Button
                                variant="outlined"
                                size="small"
                                startIcon={exporting ? <CircularProgress size={16} /> : <DownloadIcon />}
                                onClick={handleExportImage}
                                disabled={exporting || loadingContent}
                                className="border-slate-300 text-slate-600 hover:bg-slate-50 hover:border-slate-400"
                            >
                                {exporting ? t('reports.processing') : t('reports.export')}
                            </Button>
                        </Tooltip>
                    </div>
                )}
                
                <div className="flex-1 overflow-y-auto p-6 md:p-10 custom-scrollbar">
                    {selectedReport ? (
                        <div 
                            className="max-w-4xl mx-auto bg-white border border-slate-200 rounded-lg overflow-hidden min-h-[80vh] flex flex-col shadow-sm"
                            ref={reportRef}
                        >
                            {/* Report Header */}
                            <div className="p-8 border-b border-slate-100 bg-slate-50/30">
                                <div className="flex justify-between items-start mb-6">
                                    <div>
                                        <Typography variant="caption" className="text-primary-DEFAULT font-bold tracking-[0.2em] block mb-2">
                                            {selectedReport.is_summary ? t('reports.market_intelligence') : t('reports.fund_analysis')}
                                        </Typography>
                                        <Typography variant="h4" className="text-slate-900 font-extrabold tracking-tight mb-2">
                                            {selectedReport.is_summary ? t('reports.daily_overview') : selectedReport.fund_name}
                                        </Typography>
                                        {!selectedReport.is_summary && (
                                            <div className="flex items-center gap-2">
                                                <span className="font-mono text-sm text-slate-500 bg-slate-100 px-2 py-1 rounded border border-slate-200">
                                                    {selectedReport.fund_code}
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                    <Chip
                                        icon={<AccessTimeIcon style={{ fontSize: 16 }} />}
                                        label={selectedReport.date}
                                        size="small"
                                        className="bg-white border border-slate-200 text-slate-500 font-mono"
                                    />
                                </div>

                                <div className="flex gap-2">
                                    <Chip
                                        label={selectedReport.mode === 'pre' ? 'PRE-MARKET' : 'POST-MARKET'}
                                        size="small"
                                        className={`${
                                            selectedReport.mode === 'pre' 
                                            ? 'bg-blue-50 text-blue-700 border border-blue-100' 
                                            : 'bg-amber-50 text-amber-700 border border-amber-100'
                                        } font-bold tracking-wider`}
                                    />
                                </div>
                            </div>

                            {/* Report Body */}
                            <div className="p-8 md:p-12 flex-1 bg-white">
                                {loadingContent ? (
                                    <div className="flex justify-center items-center h-64">
                                        <CircularProgress size={40} thickness={4} />
                                    </div>
                                ) : (
                                    <div className="markdown-body">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{reportContent}</ReactMarkdown>
                                    </div>
                                )}
                            </div>

                            {/* Footer */}
                            <div className="p-4 bg-slate-50 border-t border-slate-100 text-center">
                                <Typography variant="caption" className="text-slate-400 font-mono text-[10px] tracking-widest uppercase">
                                    {t('reports.footer_text')} • {new Date().getFullYear()}
                                </Typography>
                            </div>
                        </div>
                    ) : (
                        <div className="flex flex-col justify-center items-center h-full text-slate-400">
                            <BusinessCenterIcon sx={{ fontSize: 60, opacity: 0.2, mb: 2 }} />
                            <Typography variant="h6" className="font-light tracking-wide">{t('reports.select_prompt')}</Typography>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}