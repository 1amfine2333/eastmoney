import { useEffect, useState } from 'react';
import {
    Box,
    Grid,
    Typography,
    CircularProgress,
    Chip,
    IconButton,
    Paper,
    useTheme,
    LinearProgress
} from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import BoltIcon from '@mui/icons-material/Bolt';
import WaterDropIcon from '@mui/icons-material/WaterDrop';
import PublicIcon from '@mui/icons-material/Public';
import StorageIcon from '@mui/icons-material/Storage';
import TimelineIcon from '@mui/icons-material/Timeline';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import { fetchDashboardOverview } from '../api';

// --- Utility Components ---

const NumberMono = ({ children, className = "", style = {} }: any) => (
    <span className={`font-mono tracking-tight ${className}`} style={{ ...style, fontVariantNumeric: 'tabular-nums' }}>
        {children}
    </span>
);

const TrendIcon = ({ val }: { val: number }) => {
    if (val > 0) return <ArrowDropUpIcon fontSize="small" className="text-red-500 -my-1" />;
    if (val < 0) return <ArrowDropDownIcon fontSize="small" className="text-green-500 -my-1" />;
    return <span className="w-4 h-4 inline-block" />;
};

const ColorVal = ({ val, suffix = "", bold = true }: { val: number, suffix?: string, bold?: boolean }) => {
    const colorClass = val > 0 ? "text-red-600" : val < 0 ? "text-green-600" : "text-slate-500";
    return (
        <NumberMono className={`${bold ? 'font-bold' : ''} ${colorClass}`}>
            {val > 0 ? '+' : ''}{val}{suffix}
        </NumberMono>
    );
};

// --- Page Component ---

export default function DashboardPage() {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    const loadData = async () => {
        if (!data) setLoading(true);
        try {
            const res = await fetchDashboardOverview();
            setData(res);
        } catch (error) {
            console.error("Failed to load dashboard", error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
        const interval = setInterval(loadData, 30000);
        return () => clearInterval(interval);
    }, []);

    if (loading && !data) return <Box className="h-screen flex items-center justify-center"><CircularProgress size={24} className="text-slate-400"/></Box>;
    if (!data) return <Box className="p-10">System Offline</Box>;

    const { market_overview, gold_macro, sectors, abnormal_movements, top_flows, system_stats } = data;
    const breadth = market_overview?.breadth || { up: 0, down: 0, flat: 0, limit_up: 0, limit_down: 0 };
    const total_stocks = breadth.up + breadth.down + breadth.flat || 1;
    const up_pct = (breadth.up / total_stocks) * 100;

    return (
        <Box className="flex flex-col gap-6 w-full h-full pb-10">
            
            {/* Header */}
            <Box className="flex justify-between items-center">
                <Box className="flex items-center gap-3">
                    <Typography variant="h5" className="font-extrabold text-slate-800 tracking-tight">
                        Market Overview
                    </Typography>
                    <Chip 
                        label="LIVE" 
                        size="small" 
                        color="success" 
                        className="h-5 text-[10px] font-bold"
                    />
                </Box>
                <IconButton size="small" onClick={loadData} className="bg-white border border-slate-200 shadow-sm hover:bg-slate-50">
                    <RefreshIcon fontSize="small" />
                </IconButton>
            </Box>

            {/* LAYER 1: MACRO STRIP (Flex Row - Full Width) */}
            <Paper elevation={0} className="w-full border border-slate-200 rounded-xl bg-white flex flex-col lg:flex-row overflow-hidden shadow-sm min-h-[140px]">
                
                {/* 1. System Status */}
                <Box className="flex-[0.8] p-5 border-b lg:border-b-0 lg:border-r border-slate-100 flex flex-col justify-between relative overflow-hidden bg-slate-50/50">
                    <StorageIcon className="absolute -right-4 -bottom-4 text-slate-200 opacity-20 text-8xl" />
                    <Box>
                        <Typography variant="caption" className="text-slate-400 font-bold uppercase tracking-wider">System Reports</Typography>
                        <Box className="flex items-baseline gap-2 mt-1">
                             <Typography variant="h3" className="font-bold text-slate-800 leading-none">
                                {system_stats?.total}
                            </Typography>
                             <Typography variant="caption" className="text-slate-400 font-bold">Files</Typography>
                        </Box>
                    </Box>
                    <Box className="flex gap-2 mt-2">
                        <Chip label={`Pre ${system_stats?.breakdown?.pre}`} size="small" className="h-5 text-[10px] bg-indigo-100 text-indigo-700 font-bold border border-indigo-200" />
                        <Chip label={`Post ${system_stats?.breakdown?.post}`} size="small" className="h-5 text-[10px] bg-purple-100 text-purple-700 font-bold border border-purple-200" />
                    </Box>
                </Box>

                {/* 2. Market Indices (Expandable) */}
                <Box className="flex-[2] flex flex-col sm:flex-row border-b lg:border-b-0 lg:border-r border-slate-100">
                    {market_overview?.indices?.map((idx: any, i: number) => (
                        <Box key={idx.name} className={`flex-1 p-5 flex flex-col justify-center hover:bg-slate-50 transition-colors ${i > 0 ? 'border-t sm:border-t-0 sm:border-l border-slate-100' : ''}`}>
                            <Typography variant="caption" className="text-slate-400 font-bold uppercase text-[10px] truncate">{idx.name}</Typography>
                            <Typography variant="h5" className="font-bold text-slate-800 leading-tight my-2">
                                <NumberMono>{idx.price}</NumberMono>
                            </Typography>
                            <Box className="flex items-center gap-2">
                                <span className={`text-sm font-bold px-1.5 py-0.5 rounded ${idx.change >= 0 ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600'}`}>
                                    {idx.change >= 0 ? '+' : ''}{idx.change}%
                                </span>
                            </Box>
                        </Box>
                    ))}
                </Box>

                {/* 3. Sentiment & Turnover */}
                <Box className="flex-[1.5] p-5 border-b lg:border-b-0 lg:border-r border-slate-100 flex flex-col justify-between">
                     <Box className="flex justify-between items-start mb-2">
                        <Box>
                             <Typography variant="caption" className="text-slate-400 font-bold uppercase block text-[10px]">Market Sentiment</Typography>
                             <Box className="flex items-baseline gap-1">
                                <span className="text-red-500 font-bold text-sm">{breadth.up} Up</span>
                                <span className="text-slate-300 text-xs">/</span>
                                <span className="text-green-500 font-bold text-sm">{breadth.down} Down</span>
                             </Box>
                        </Box>
                        <Box className="text-right">
                             <Typography variant="caption" className="text-slate-400 font-bold uppercase block text-[10px]">Turnover</Typography>
                             <Typography variant="subtitle1" className="font-bold text-slate-800 leading-none">
                                {market_overview?.turnover?.total}<span className="text-xs font-normal text-slate-400">亿</span>
                            </Typography>
                        </Box>
                     </Box>
                     
                     <Box className="w-full h-2 bg-green-100 rounded-full overflow-hidden flex mb-2">
                        <div className="h-full bg-red-500" style={{ width: `${up_pct}%` }}></div>
                        <div className="h-full bg-green-500 flex-1"></div>
                    </Box>
                    
                    <Box className="flex gap-2 mt-1">
                        <span className="flex-1 text-center text-[10px] font-bold bg-red-50 text-red-600 py-0.5 rounded border border-red-100">
                            Limit Up {breadth.limit_up}
                        </span>
                        <span className="flex-1 text-center text-[10px] font-bold bg-green-50 text-green-600 py-0.5 rounded border border-green-100">
                            Limit Down {breadth.limit_down}
                        </span>
                    </Box>
                </Box>

                {/* 4. Gold (Light Mode) */}
                <Box className="flex-[0.8] p-5 bg-white border-l border-slate-100 flex flex-col justify-center relative overflow-hidden">
                    <PublicIcon className="absolute -right-6 -bottom-8 text-amber-500 opacity-10 text-[120px]" />
                    <Box className="relative z-10">
                        <Box className="flex justify-between items-center mb-1">
                            <Typography variant="caption" className="text-slate-400 font-bold uppercase tracking-wider">Gold</Typography>
                            <Typography variant="caption" className="text-amber-600 font-bold bg-amber-50 px-1 rounded">XAU</Typography>
                        </Box>
                        <Typography variant="h4" className="font-mono font-bold text-slate-800 tracking-tight">
                            ${gold_macro?.price}
                        </Typography>
                        <Box className="flex items-center gap-2 mt-1">
                             <span className={`text-xs font-bold ${gold_macro?.change_pct >= 0 ? 'text-red-600' : 'text-green-600'}`}>
                                {gold_macro?.change_pct >= 0 ? '+' : ''}{gold_macro?.change_pct}%
                            </span>
                            <span className="text-[10px] text-slate-400 font-mono font-bold bg-slate-50 px-1 rounded border border-slate-100">DXY {gold_macro?.dxy}</span>
                        </Box>
                    </Box>
                </Box>
            </Paper>


            {/* LAYER 2: DEEP DIVE (Horizontal Split, Auto Height) */}
            <Box className="flex flex-col lg:flex-row gap-6 w-full">
                
                {/* Left: Smart Money (Full List, No Scroll) */}
                <Paper elevation={0} className="flex-[1.4] border border-slate-200 rounded-xl bg-white flex flex-col shadow-sm">
                    <Box className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/30">
                        <Box className="flex items-center gap-3">
                            <Box className="p-2 bg-indigo-50 rounded-lg">
                                <TimelineIcon className="text-indigo-600" fontSize="small"/>
                            </Box>
                            <Box>
                                <Typography variant="subtitle1" className="font-bold text-slate-800 leading-tight">Smart Money Flow</Typography>
                                <Typography variant="caption" className="text-slate-400 font-medium">Top Net Inflow Leaders</Typography>
                            </Box>
                        </Box>
                        <Chip label={`Main Flow: ${market_overview?.main_flow}亿`} size="small" className="bg-indigo-50 text-indigo-700 font-bold border border-indigo-100" />
                    </Box>
                    <Box className="flex-1">
                        <table className="w-full text-sm text-left">
                            <thead className="bg-white text-slate-400 font-medium text-[11px] uppercase tracking-wider border-b border-slate-50">
                                <tr>
                                    <th className="px-6 py-4 font-bold bg-slate-50/20">Code</th>
                                    <th className="px-6 py-4 font-bold bg-slate-50/20">Name</th>
                                    <th className="px-6 py-4 font-bold bg-slate-50/20 text-right">Net Inflow (亿)</th>
                                    <th className="px-6 py-4 font-bold bg-slate-50/20 text-right">Change</th>
                                    <th className="px-6 py-4 font-bold bg-slate-50/20 text-right w-24">Trend</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-50">
                                {top_flows?.map((row: any) => (
                                    <tr key={row.code} className="hover:bg-indigo-50/5 transition-colors group">
                                        <td className="px-6 py-4 font-mono text-slate-400 group-hover:text-indigo-500 transition-colors">{row.code}</td>
                                        <td className="px-6 py-4 font-bold text-slate-700 text-base">{row.name}</td>
                                        <td className="px-6 py-4 text-right">
                                            <NumberMono className="text-indigo-700 font-bold bg-indigo-50 px-2 py-1 rounded-md">+{row.net_buy}</NumberMono>
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <ColorVal val={row.change_pct} suffix="%" />
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden ml-auto">
                                                <div className="h-full bg-red-500" style={{ width: `${Math.min(Math.abs(row.change_pct) * 5, 100)}%` }}></div>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </Box>
                </Paper>

                {/* Right: Sector Performance (Stacked, No Scroll) */}
                <Box className="flex-1 flex flex-col gap-4">
                     {/* Gainers */}
                    <Paper elevation={0} className="border border-slate-200 rounded-xl bg-white overflow-hidden flex flex-col shadow-sm h-full">
                        <Box className="px-5 py-3 border-b border-red-50 bg-red-50/20 flex justify-between items-center">
                            <Box className="flex items-center gap-2">
                                <ArrowDropUpIcon className="text-red-500" />
                                <Typography variant="subtitle2" className="font-bold text-slate-700">Sector Leaders</Typography>
                            </Box>
                            <Typography variant="caption" className="text-red-400 font-bold text-[10px] uppercase tracking-wide">Top 5 Gainers</Typography>
                        </Box>
                        <Box className="flex-1">
                            {sectors?.gainers?.slice(0, 5).map((s: any, i: number) => (
                                <Box key={s.name} className="flex justify-between items-center px-5 py-4 border-b border-slate-50 last:border-0 hover:bg-red-50/5 transition-colors">
                                    <Box className="flex items-center gap-3">
                                        <span className="text-[10px] font-mono text-white bg-red-400 w-5 h-5 flex items-center justify-center rounded font-bold shadow-sm shadow-red-200">{i+1}</span>
                                        <Typography variant="body2" className="font-bold text-slate-700">{s.name}</Typography>
                                    </Box>
                                    <ColorVal val={s.change} suffix="%" />
                                </Box>
                            ))}
                        </Box>
                    </Paper>

                    {/* Losers */}
                    <Paper elevation={0} className="border border-slate-200 rounded-xl bg-white overflow-hidden flex flex-col shadow-sm h-full">
                        <Box className="px-5 py-3 border-b border-green-50 bg-green-50/20 flex justify-between items-center">
                            <Box className="flex items-center gap-2">
                                <ArrowDropDownIcon className="text-green-500" />
                                <Typography variant="subtitle2" className="font-bold text-slate-700">Sector Laggards</Typography>
                            </Box>
                             <Typography variant="caption" className="text-green-400 font-bold text-[10px] uppercase tracking-wide">Top 5 Losers</Typography>
                        </Box>
                        <Box className="flex-1">
                            {sectors?.losers?.slice().reverse().slice(0, 5).map((s: any, i: number) => (
                                <Box key={s.name} className="flex justify-between items-center px-5 py-4 border-b border-slate-50 last:border-0 hover:bg-green-50/5 transition-colors">
                                    <Box className="flex items-center gap-3">
                                        <span className="text-[10px] font-mono text-white bg-green-400 w-5 h-5 flex items-center justify-center rounded font-bold shadow-sm shadow-green-200">{i+1}</span>
                                        <Typography variant="body2" className="font-bold text-slate-700">{s.name}</Typography>
                                    </Box>
                                    <ColorVal val={s.change} suffix="%" />
                                </Box>
                            ))}
                        </Box>
                    </Paper>
                </Box>
            </Box>

            {/* LAYER 3: LIVE SIGNALS */}
            <Paper elevation={0} className="border border-slate-200 rounded-xl bg-white overflow-hidden flex flex-col shadow-sm min-h-[160px]">
                <Box className="px-6 py-3 border-b border-slate-100 bg-amber-50/10 flex justify-between items-center">
                    <Box className="flex items-center gap-2">
                        <BoltIcon className="text-amber-500" />
                        <Typography variant="subtitle1" className="font-bold text-slate-700">Live Market Signals</Typography>
                    </Box>
                    <Box className="flex items-center gap-2">
                        <span className="animate-pulse w-2 h-2 rounded-full bg-amber-500"></span>
                        <Typography variant="caption" className="text-slate-400 font-mono text-[10px]">REAL-TIME</Typography>
                    </Box>
                </Box>
                
                <Box className="p-5 overflow-x-auto whitespace-nowrap scrollbar-hide flex gap-4 items-center bg-slate-50/30 flex-1">
                    {abnormal_movements?.map((m: any, i: number) => (
                        <Paper key={i} elevation={0} className="inline-block min-w-[220px] p-4 rounded-xl border border-slate-200 bg-white hover:border-amber-300 hover:shadow-md transition-all relative group flex-shrink-0 cursor-default">
                            <Box className="flex justify-between items-start mb-2">
                                <span className="bg-slate-100 text-slate-400 px-2 py-0.5 rounded text-[10px] font-mono font-bold">
                                    {m.time}
                                </span>
                                {m.type.includes('拉升') || m.type.includes('涨') ? (
                                    <ArrowDropUpIcon fontSize="small" className="text-red-500 bg-red-50 rounded-full p-0.5"/>
                                ) : (
                                    <ArrowDropDownIcon fontSize="small" className="text-green-500 bg-green-50 rounded-full p-0.5"/>
                                )}
                            </Box>
                            
                            <Box className="flex items-baseline gap-2 mb-1">
                                <Typography variant="subtitle1" className="font-bold text-slate-800 truncate">
                                    {m.name}
                                </Typography>
                            </Box>
                            <Typography variant="caption" className="text-slate-400 font-mono text-[10px] block mb-2">
                                {m.info.replace('Code: ', '')}
                            </Typography>
                            
                            <Box>
                                <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wide border ${
                                    m.type.includes('拉升') || m.type.includes('涨') ? 'bg-red-50 text-red-600 border-red-100' : 
                                    m.type.includes('跳水') || m.type.includes('跌') ? 'bg-green-50 text-green-600 border-green-100' : 'bg-slate-100 text-slate-500 border-slate-200'
                                }`}>
                                    {m.type.replace('Sector', '').trim()}
                                </span>
                            </Box>
                        </Paper>
                    ))}
                    
                    {(!abnormal_movements || abnormal_movements.length === 0) && (
                        <Box className="w-full text-center text-slate-400 text-sm italic">
                            Waiting for market signals...
                        </Box>
                    )}
                </Box>
            </Paper>

        </Box>
    );
}