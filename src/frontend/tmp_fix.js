// 临时修复函数
function fixedDisplayAnalysis(structuredResponse) {
    var analysisContent = document.getElementById('analysis-content');
    
    if (!structuredResponse || Object.keys(structuredResponse).length === 0) {
        analysisContent.innerHTML = '<div class="analysis-empty">暂无分析过程</div>';
        return;
    }
    
    var htmlContent = '';
    
    // 显示分步分析
    if (structuredResponse.step_by_step_analysis) {
        htmlContent += '<div class="analysis-item">' +
            '<div class="analysis-title">分步分析 (Step-by-step Analysis)</div>' +
            '<div class="analysis-data">' + structuredResponse.step_by_step_analysis.replace(/\n/g, '<br>') + '</div>' +
        '</div>';
    }
    
    // 显示推理摘要
    if (structuredResponse.reasoning_summary) {
        htmlContent += '<div class="analysis-item">' +
            '<div class="analysis-title">推理摘要 (Reasoning Summary)</div>' +
            '<div class="analysis-data">' + structuredResponse.reasoning_summary + '</div>' +
        '</div>';
    }
    
    // 显示相关页面（如果存在原始搜索结果）
    if (window.currentSearchResults && structuredResponse.relevant_pages && structuredResponse.relevant_pages.length > 0) {
        var pages = structuredResponse.relevant_pages;
        var pageDetails = [];
        
        // 遍历相关页面列表
        for (var i = 0; i < pages.length; i++) {
            var page = pages[i];
            var found = false;
            
            // 在搜索结果中查找对应的文档
            for (var j = 0; j < window.currentSearchResults.length; j++) {
                var result = window.currentSearchResults[j];
                
                // 检查是否有页码信息
                if (result.metadata && result.metadata.page == page) {
                    var fileName = result.metadata.filename || '未知文件';
                    pageDetails.push('第'+page+'页 ('+fileName+')');
                    found = true;
                    break;
                }
            }
            
            // 如果没找到，尝试从内容中查找页面信息
            if (!found) {
                for (var j = 0; j < window.currentSearchResults.length; j++) {
                    var result = window.currentSearchResults[j];
                    
                    // 检查内容中是否包含页面号
                    if (result.content && (result.content.indexOf('/ ' + page + ' /') !== -1 || 
                                          result.content.indexOf('第 ' + page + ' 页') !== -1 ||
                                          result.content.indexOf(page + ' / ') !== -1)) {
                        var fileName = result.metadata && result.metadata.filename ? result.metadata.filename : '未知文件';
                        pageDetails.push('第'+page+'页 ('+fileName+')');
                        found = true;
                        break;
                    }
                }
            }
            
            // 如果还是没找到，就只显示页码
            if (!found) {
                pageDetails.push('第'+page+'页');
            }
        }
        
        htmlContent += '<div class="analysis-item">' +
            '<div class="analysis-title">相关页面 (Relevant Pages)</div>' +
            '<div class="analysis-data">' + pageDetails.join(', ') + '</div>' +
        '</div>';
    } else if (structuredResponse.relevant_pages && structuredResponse.relevant_pages.length > 0) {
        var pagesStr = structuredResponse.relevant_pages.join(', ');
        htmlContent += '<div class="analysis-item">' +
            '<div class="analysis-title">相关页面 (Relevant Pages)</div>' +
            '<div class="analysis-data">第 ' + pagesStr + ' 页</div>' +
        '</div>';
    }
    
    if (htmlContent === '') {
        htmlContent = '<div class="analysis-empty">暂无分析过程</div>';
    }
    
    analysisContent.innerHTML = htmlContent;
}