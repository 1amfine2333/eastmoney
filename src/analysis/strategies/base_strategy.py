from abc import ABC, abstractmethod
from typing import Dict, Any

class AnalysisStrategy(ABC):
    """
    Abstract Base Class for Fund Analysis Strategies.
    """
    
    def __init__(self, fund_info: Dict[str, Any], llm_client, web_search):
        self.fund_info = fund_info
        self.fund_code = fund_info.get("code")
        self.fund_name = fund_info.get("name")
        self.llm = llm_client
        self.web_search = web_search
        self.sources = [] # List to track sources

    def _add_source(self, category: str, title: str, url: str, source_name: str = "Web"):
        """Add a source to the tracking list."""
        self.sources.append({
            "category": category,
            "title": title,
            "url": url,
            "source": source_name
        })

    def get_sources(self) -> str:
        """Format collected sources."""
        if not self.sources:
            return ""
        
        output = ["\n\n## 📚 引用来源 (Reference Sources)"]
        unique_sources = {}
        
        # Deduplicate based on URL
        for s in self.sources:
            if s['url'] and s['url'] not in unique_sources:
                unique_sources[s['url']] = s
        
        for idx, s in enumerate(unique_sources.values(), 1):
            output.append(f"{idx}. [{s['category']}] [{s['title']}]({s['url']})")
            
        return "\n".join(output)

    @abstractmethod
    def collect_data(self, mode: str) -> Dict[str, Any]:
        """
        Collect data relevant to the specific strategy.
        mode: 'pre' or 'post'
        """
        pass

    @abstractmethod
    def generate_report(self, mode: str, data: Dict[str, Any]) -> str:
        """
        Generate the analysis report using the collected data.
        mode: 'pre' or 'post'
        """
        pass
