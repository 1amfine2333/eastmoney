import { useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  Typography,
  Button,
  CircularProgress,
} from '@mui/material';
import ArticleIcon from '@mui/icons-material/Article';

import NewsCard from './NewsCard';
import type { NewsItem } from '../../api';

interface NewsFeedProps {
  news: NewsItem[];
  selectedId?: string;
  onSelect: (item: NewsItem) => void;
  onToggleBookmark: (item: NewsItem) => void;
  onLoadMore: () => void;
  hasMore: boolean;
  loading: boolean;
}

export default function NewsFeed({
  news,
  selectedId,
  onSelect,
  onToggleBookmark,
  onLoadMore,
  hasMore,
  loading,
}: NewsFeedProps) {
  const { t } = useTranslation();
  const observerRef = useRef<IntersectionObserver | null>(null);

  // Infinite scroll using Intersection Observer
  const lastNewsRef = useCallback((node: HTMLDivElement | null) => {
    if (loading) return;

    if (observerRef.current) {
      observerRef.current.disconnect();
    }

    observerRef.current = new IntersectionObserver(entries => {
      if (entries[0].isIntersecting && hasMore) {
        onLoadMore();
      }
    });

    if (node) {
      observerRef.current.observe(node);
    }
  }, [loading, hasMore, onLoadMore]);

  if (news.length === 0 && !loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          p: 4,
          color: '#94a3b8',
        }}
      >
        <ArticleIcon sx={{ fontSize: 64, mb: 2, opacity: 0.5 }} />
        <Typography variant="h6" sx={{ mb: 1 }}>
          {t('news.feed.no_news')}
        </Typography>
        <Typography variant="body2" sx={{ textAlign: 'center' }}>
          {t('news.feed.no_news_hint')}
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ position: 'relative' }}>
      {news.map((item, index) => {
        const isLast = index === news.length - 1;
        return (
          <div key={item.id} ref={isLast ? lastNewsRef : undefined}>
            <NewsCard
              news={item}
              selected={selectedId === item.id}
              onClick={() => onSelect(item)}
              onToggleBookmark={(e) => {
                e.stopPropagation();
                onToggleBookmark(item);
              }}
            />
          </div>
        );
      })}

      {/* Loading indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
          <CircularProgress size={24} />
        </Box>
      )}

      {/* Load more button (fallback) */}
      {hasMore && !loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
          <Button
            variant="outlined"
            onClick={onLoadMore}
            sx={{
              borderColor: '#e2e8f0',
              color: '#64748b',
              '&:hover': {
                borderColor: '#cbd5e1',
                backgroundColor: '#f8fafc',
              },
            }}
          >
            {t('news.feed.load_more')}
          </Button>
        </Box>
      )}

      {/* End of list */}
      {!hasMore && news.length > 0 && (
        <Box sx={{ textAlign: 'center', p: 3, color: '#94a3b8' }}>
          <Typography variant="body2">
            {t('news.feed.all_loaded')}
          </Typography>
        </Box>
      )}
    </Box>
  );
}
