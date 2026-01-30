import { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Box,
  Typography,
  CircularProgress,
  Snackbar,
  Alert,
  useMediaQuery,
  useTheme,
} from '@mui/material';

import NewsSidebar from '../components/news/NewsSidebar';
import NewsFeed from '../components/news/NewsFeed';
import NewsDetail from '../components/news/NewsDetail';
import NewsFilter from '../components/news/NewsFilter';

import { fetchNewsFeed, fetchNewsDetail, toggleNewsBookmark, fetchNewsBookmarks } from '../api';
import type { NewsItem, NewsFeedResponse } from '../api';

export type NewsCategory = 'all' | 'realtime' | 'morning' | 'flash' | 'announcement' | 'research' | 'hot';

const NEWS_PAGE_SIZE = 50;

type TimeRange = 'all' | '1d' | '3d' | '7d';

const parsePublishedAtMs = (value?: string) => {
  if (!value) return 0;
  const s = String(value).trim().replace('T', ' ').replace('Z', '');

  // YYYYMMDD HH:mm:ss
  if (/^\d{8}\s\d{2}:\d{2}:\d{2}/.test(s)) {
    const y = s.slice(0, 4);
    const m = s.slice(4, 6);
    const d = s.slice(6, 8);
    const time = s.slice(9, 17);
    const t = Date.parse(`${y}-${m}-${d}T${time}`);
    return Number.isFinite(t) ? t : 0;
  }

  // YYYYMMDD
  if (/^\d{8}$/.test(s)) {
    const y = s.slice(0, 4);
    const m = s.slice(4, 6);
    const d = s.slice(6, 8);
    const t = Date.parse(`${y}-${m}-${d}`);
    return Number.isFinite(t) ? t : 0;
  }

  // YYYY-MM-DD HH:mm:ss / YYYY-MM-DD
  const normalized = s.includes(' ') ? s.replace(' ', 'T') : s;
  const t = Date.parse(normalized);
  return Number.isFinite(t) ? t : 0;
};

export default function NewsPage() {
  const { t } = useTranslation();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));

  // News state
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Pagination
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);
  const [total, setTotal] = useState(0);

  // Category & Filter
  const [category, setCategory] = useState<NewsCategory>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [timeRange, setTimeRange] = useState<TimeRange>('3d');

  // Selected news for detail view
  const [selectedNews, setSelectedNews] = useState<NewsItem | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [newsAnalysis, setNewsAnalysis] = useState<any>(null);

  // Watchlist info
  const [hasWatchlist, setHasWatchlist] = useState(false);
  const [watchlistSummary, setWatchlistSummary] = useState<any>(null);

  // Bookmarks
  const [bookmarks, setBookmarks] = useState<NewsItem[]>([]);
  const [showBookmarks, setShowBookmarks] = useState(false);

  // Notification
  const [notify, setNotify] = useState<{ open: boolean; message: string; severity: 'success' | 'error' | 'info' }>({
    open: false,
    message: '',
    severity: 'info',
  });

  const showNotify = (message: string, severity: 'success' | 'error' | 'info' = 'info') => {
    setNotify({ open: true, message, severity });
  };

  // Fetch news feed
  const loadNews = useCallback(async (resetPage = false) => {
    try {
      setLoading(true);
      setError(null);

      const currentPage = resetPage ? 1 : page;
      const sinceDays = timeRange === 'all' ? undefined : Number(timeRange.replace('d', ''));
      const response: NewsFeedResponse = await fetchNewsFeed(category, currentPage, NEWS_PAGE_SIZE, sinceDays);

      if (resetPage) {
        setNews(response.news || []);
        setPage(1);
      } else {
        setNews(prev => currentPage === 1 ? (response.news || []) : [...prev, ...(response.news || [])]);
      }

      setHasMore(response.has_more || false);
      setTotal(response.total || 0);
      setHasWatchlist(response.has_watchlist || false);
      setWatchlistSummary(response.watchlist_summary);
    } catch (err: any) {
      console.error('Failed to load news:', err);
      setError(err.message || 'Failed to load news');
    } finally {
      setLoading(false);
    }
  }, [category, page]);

  // Initial load and category change
  useEffect(() => {
    // Don't reload news when in bookmarks view - bookmarks are filtered client-side
    if (showBookmarks) return;
    
    setNews([]);
    setPage(1);
    setSelectedNews(null);
    loadNews(true);
  }, [category, timeRange, showBookmarks]);

  // Load more on page change
  useEffect(() => {
    if (page > 1) {
      loadNews(false);
    }
  }, [page]);

  // Load bookmarks
  const loadBookmarks = async () => {
    try {
      const response = await fetchNewsBookmarks();
      setBookmarks(response.bookmarks || []);
    } catch (err) {
      console.error('Failed to load bookmarks:', err);
    }
  };

  useEffect(() => {
    loadBookmarks();
  }, []);

  // Handle news selection
  const handleSelectNews = async (item: NewsItem) => {
    setSelectedNews(item);
    setDetailLoading(true);
    setNewsAnalysis(null);

    try {
      const response = await fetchNewsDetail(item.id, item.title, item.content || '');
      setNewsAnalysis(response.analysis);

      // Mark as read in local state
      setNews(prev => prev.map(n => n.id === item.id ? { ...n, is_read: true } : n));
    } catch (err) {
      console.error('Failed to load news detail:', err);
    } finally {
      setDetailLoading(false);
    }
  };

  // Handle bookmark toggle
  const handleToggleBookmark = async (item: NewsItem) => {
    try {
      const response = await toggleNewsBookmark(item.id, {
        news_title: item.title,
        news_source: item.source,
        news_url: item.url,
        news_category: item.category,
      });

      const newBookmarked = response.bookmarked;

      // Update local state
      setNews(prev => prev.map(n => n.id === item.id ? { ...n, is_bookmarked: newBookmarked } : n));

      if (selectedNews?.id === item.id) {
        setSelectedNews({ ...selectedNews, is_bookmarked: newBookmarked });
      }

      // Refresh bookmarks
      loadBookmarks();

      showNotify(
        newBookmarked
          ? t('news.toast.bookmarked')
          : t('news.toast.unbookmarked'),
        'success'
      );
    } catch (err) {
      console.error('Failed to toggle bookmark:', err);
      showNotify(t('news.toast.action_failed'), 'error');
    }
  };

  // Handle load more
  const handleLoadMore = () => {
    if (!loading && hasMore) {
      setPage(prev => prev + 1);
    }
  };

  // Handle category change
  const handleCategoryChange = (newCategory: NewsCategory) => {
    setCategory(newCategory);
    setShowBookmarks(false);
  };

  // Handle show bookmarks
  const handleShowBookmarks = () => {
    setShowBookmarks(true);
    setSelectedNews(null);
    loadBookmarks();  // Refresh bookmarks when switching to bookmarks view
  };

  // Filtered news for display
  const displayedNews = showBookmarks ? bookmarks : news;
  const filteredNews = searchQuery
    ? displayedNews.filter(item =>
        item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (item.content && item.content.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : displayedNews;

  const rangeFilteredNews = (() => {
    if (timeRange === 'all') return filteredNews;
    const days = Number(timeRange.replace('d', ''));
    const cutoffMs = Date.now() - days * 24 * 60 * 60 * 1000;
    return filteredNews.filter((item) => parsePublishedAtMs(item.published_at) >= cutoffMs);
  })();

  // Ensure newest news shows first across all tabs/categories
  const sortedNews = [...rangeFilteredNews].sort((a, b) => parsePublishedAtMs(b.published_at) - parsePublishedAtMs(a.published_at));

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: '1px solid #f1f5f9' }}>
        <Typography variant="h5" sx={{ fontWeight: 700, color: '#0f172a' }}>
          {t('news.title')}
        </Typography>
        <Typography variant="body2" sx={{ color: '#64748b', mt: 0.5 }}>
          {hasWatchlist
            ? t('news.header.watchlist', {
                stocksCount: watchlistSummary?.stocks_count || 0,
                fundsCount: watchlistSummary?.funds_count || 0,
              })
            : t('news.header.no_watchlist')}
        </Typography>
        {total > 0 && (
          <Typography variant="caption" sx={{ color: '#94a3b8', display: 'block', mt: 0.25 }}>
            {t('news.header.total', { count: total })}
          </Typography>
        )}
      </Box>

      {/* Main content - 3 column layout */}
      <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
        {/* Left sidebar - Categories */}
        {!isMobile && (
          <Box sx={{ width: 200, borderRight: '1px solid #f1f5f9', flexShrink: 0 }}>
            <NewsSidebar
              category={category}
              onCategoryChange={handleCategoryChange}
              showBookmarks={showBookmarks}
              onShowBookmarks={handleShowBookmarks}
              bookmarksCount={bookmarks.length}
            />
          </Box>
        )}

        {/* Center - News feed */}
        <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
          {/* Filter bar */}
          <NewsFilter
            searchQuery={searchQuery}
            onSearchChange={setSearchQuery}
            category={category}
            onCategoryChange={handleCategoryChange}
            timeRange={timeRange}
            onTimeRangeChange={(next) => {
              setTimeRange(next);
              // Don't exit bookmarks view when changing time range
            }}
            showCategoryChips={isMobile}
          />

          {/* News list */}
          <Box sx={{ flex: 1, overflow: 'auto' }}>
            {loading && news.length === 0 ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress />
              </Box>
            ) : error ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', p: 3 }}>
                <Typography color="error">{error}</Typography>
              </Box>
            ) : (
              <NewsFeed
                news={sortedNews}
                selectedId={selectedNews?.id}
                onSelect={handleSelectNews}
                onToggleBookmark={handleToggleBookmark}
                onLoadMore={handleLoadMore}
                hasMore={hasMore && !showBookmarks}
                loading={loading}
              />
            )}
          </Box>
        </Box>

        {/* Right panel - Detail view */}
        {!isMobile && (
          <Box sx={{ width: 400, borderLeft: '1px solid #f1f5f9', flexShrink: 0, overflow: 'auto' }}>
            <NewsDetail
              news={selectedNews}
              analysis={newsAnalysis}
              loading={detailLoading}
              onToggleBookmark={handleToggleBookmark}
            />
          </Box>
        )}
      </Box>

      {/* Mobile detail modal would go here */}

      {/* Notification */}
      <Snackbar
        open={notify.open}
        autoHideDuration={3000}
        onClose={() => setNotify({ ...notify, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity={notify.severity} onClose={() => setNotify({ ...notify, open: false })}>
          {notify.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}
