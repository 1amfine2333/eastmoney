import { useTranslation } from 'react-i18next';
import {
  Box,
  Typography,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import BookmarkBorderIcon from '@mui/icons-material/BookmarkBorder';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';

import type { NewsItem } from '../../api';

interface NewsCardProps {
  news: NewsItem;
  selected?: boolean;
  onClick: () => void;
  onToggleBookmark: (e: React.MouseEvent) => void;
}

const sourceColors: Record<string, 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info'> = {
  '财联社': 'error',
  'sina': 'warning',
  '新浪': 'warning',
  '东方财富': 'primary',
  '同花顺': 'secondary',
  'tushare': 'info',
  'akshare': 'success',
  '公告': 'error',
};

function formatRelativeTime(datetime: string, isZh: boolean): string {
  if (!datetime) return '';

  try {
    // Parse various datetime formats
    let date: Date;
    if (datetime.includes('T')) {
      date = new Date(datetime);
    } else if (datetime.includes(' ')) {
      date = new Date(datetime.replace(' ', 'T'));
    } else if (datetime.length === 8) {
      // YYYYMMDD format
      date = new Date(`${datetime.slice(0, 4)}-${datetime.slice(4, 6)}-${datetime.slice(6, 8)}`);
    } else {
      date = new Date(datetime);
    }

    if (isNaN(date.getTime())) return datetime;

    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return isZh ? '刚刚' : 'Just now';
    if (diffMins < 60) return isZh ? `${diffMins}分钟前` : `${diffMins}m ago`;
    if (diffHours < 24) return isZh ? `${diffHours}小时前` : `${diffHours}h ago`;
    if (diffDays < 7) return isZh ? `${diffDays}天前` : `${diffDays}d ago`;

    return date.toLocaleDateString();
  } catch {
    return datetime;
  }
}

export default function NewsCard({ news, selected, onClick, onToggleBookmark }: NewsCardProps) {
  const { t, i18n } = useTranslation();
  const isZh = i18n.language === 'zh';
  const sourceColor = sourceColors[news.source_name] || sourceColors[news.source] || 'default';

  const getSentimentIcon = () => {
    if (!news.sentiment) return null;
    switch (news.sentiment) {
      case 'positive':
        return <TrendingUpIcon sx={{ fontSize: 16, color: '#22c55e' }} />;
      case 'negative':
        return <TrendingDownIcon sx={{ fontSize: 16, color: '#ef4444' }} />;
      default:
        return <TrendingFlatIcon sx={{ fontSize: 16, color: '#64748b' }} />;
    }
  };

  return (
    <Box
      onClick={onClick}
      sx={{
        p: 2,
        cursor: 'pointer',
        borderBottom: '1px solid #f1f5f9',
        backgroundColor: selected ? '#f8fafc' : (news.is_read ? '#fafafa' : '#fff'),
        transition: 'background-color 0.15s',
        '&:hover': {
          backgroundColor: '#f1f5f9',
        },
        opacity: news.is_read ? 0.8 : 1,
      }}
    >
      {/* Header: Source + Time + Bookmark */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Chip
          label={news.source_name || news.source}
          size="small"
          color={sourceColor}
          sx={{ height: 20, fontSize: '0.7rem' }}
        />
        {news.importance === 'high' && (
          <Chip
            label={t('news.card.important')}
            size="small"
            color="error"
            variant="outlined"
            sx={{ height: 20, fontSize: '0.7rem' }}
          />
        )}
        {getSentimentIcon()}
        <Typography variant="caption" sx={{ color: '#94a3b8', ml: 'auto' }}>
          {formatRelativeTime(news.published_at, isZh)}
        </Typography>
        <Tooltip title={news.is_bookmarked ? t('news.card.unbookmark') : t('news.card.bookmark')}>
          <IconButton
            size="small"
            onClick={onToggleBookmark}
            sx={{
              p: 0.5,
              color: news.is_bookmarked ? '#f59e0b' : '#cbd5e1',
              '&:hover': { color: '#f59e0b' },
            }}
          >
            {news.is_bookmarked ? <BookmarkIcon fontSize="small" /> : <BookmarkBorderIcon fontSize="small" />}
          </IconButton>
        </Tooltip>
      </Box>

      {/* Title */}
      <Typography
        sx={{
          fontWeight: news.is_read ? 400 : 600,
          fontSize: '0.95rem',
          color: '#0f172a',
          lineHeight: 1.4,
          display: '-webkit-box',
          WebkitLineClamp: 2,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
          mb: 0.5,
        }}
      >
        {news.title}
      </Typography>

      {/* Content preview */}
      {news.content && (
        <Typography
          variant="body2"
          sx={{
            color: '#64748b',
            lineHeight: 1.5,
            display: '-webkit-box',
            WebkitLineClamp: 2,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            fontSize: '0.85rem',
          }}
        >
          {news.summary || news.content}
        </Typography>
      )}

      {/* Related stocks */}
      {news.related_stocks && news.related_stocks.length > 0 && (
        <Box sx={{ display: 'flex', gap: 0.5, mt: 1, flexWrap: 'wrap' }}>
          {news.related_stocks.slice(0, 3).map((stock) => (
            <Chip
              key={stock.code}
              label={stock.name || stock.code}
              size="small"
              variant="outlined"
              sx={{
                height: 20,
                fontSize: '0.7rem',
                borderColor: '#e2e8f0',
                color: '#475569',
              }}
            />
          ))}
        </Box>
      )}
    </Box>
  );
}
