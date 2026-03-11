/**
 * 前端工具函数库
 * 提取公共的 HTTP 请求、DOM 操作等函数
 */

// API 基础 URL
const API_BASE_URL = 'http://localhost:8080';

/**
 * 统一的 HTTP 请求函数
 * @param {string} method - HTTP 方法 (GET, POST, etc.)
 * @param {string} url - 请求 URL
 * @param {object} data - 请求数据（可选）
 * @returns {Promise} 返回 Promise，resolve 为响应数据
 */
function makeRequest(method, url, data = null) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open(method, url, true);
        xhr.setRequestHeader('Content-Type', 'application/json');
        
        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4) {
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (e) {
                        reject(new Error('响应解析失败: ' + e.message));
                    }
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            }
        };
        
        xhr.onerror = function() {
            reject(new Error('网络请求失败'));
        };
        
        if (data) {
            xhr.send(JSON.stringify(data));
        } else {
            xhr.send();
        }
    });
}

/**
 * 上传文件（支持 FormData）
 * @param {string} url - 上传 URL
 * @param {FormData} formData - 表单数据
 * @param {function} onProgress - 进度回调函数
 * @returns {Promise} 返回 Promise
 */
function uploadFile(url, formData, onProgress = null) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('POST', url, true);
        
        // 监听上传进度
        if (onProgress && xhr.upload) {
            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    onProgress(percentComplete);
                }
            });
        }
        
        xhr.onreadystatechange = function() {
            if (xhr.readyState === 4) {
                if (xhr.status === 200) {
                    try {
                        const response = JSON.parse(xhr.responseText);
                        resolve(response);
                    } catch (e) {
                        reject(new Error('响应解析失败: ' + e.message));
                    }
                } else {
                    reject(new Error(`HTTP ${xhr.status}: ${xhr.statusText}`));
                }
            }
        };
        
        xhr.onerror = function() {
            reject(new Error('上传失败'));
        };
        
        xhr.send(formData);
    });
}

/**
 * 显示加载状态
 * @param {HTMLElement} element - 按钮元素
 * @param {string} loadingText - 加载文本
 */
function showLoading(element, loadingText = '加载中...') {
    element.disabled = true;
    element.classList.add('loading');
    element.dataset.originalText = element.textContent;
    element.textContent = loadingText;
}

/**
 * 隐藏加载状态
 * @param {HTMLElement} element - 按钮元素
 */
function hideLoading(element) {
    element.disabled = false;
    element.classList.remove('loading');
    element.textContent = element.dataset.originalText || '发送';
}

/**
 * 显示错误提示
 * @param {string} message - 错误信息
 */
function showError(message) {
    console.error(message);
    alert('错误: ' + message);
}

/**
 * 显示成功提示
 * @param {string} message - 成功信息
 */
function showSuccess(message) {
    console.log(message);
    alert('成功: ' + message);
}

/**
 * 防抖函数
 * @param {function} func - 要防抖的函数
 * @param {number} delay - 延迟时间（毫秒）
 * @returns {function} 防抖后的函数
 */
function debounce(func, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

/**
 * 节流函数
 * @param {function} func - 要节流的函数
 * @param {number} interval - 时间间隔（毫秒）
 * @returns {function} 节流后的函数
 */
function throttle(func, interval) {
    let lastTime = 0;
    return function(...args) {
        const now = Date.now();
        if (now - lastTime >= interval) {
            func.apply(this, args);
            lastTime = now;
        }
    };
}

/**
 * 格式化时间
 * @param {number} seconds - 秒数
 * @returns {string} 格式化后的时间字符串
 */
function formatTime(seconds) {
    if (seconds < 60) {
        return seconds.toFixed(2) + '秒';
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const secs = (seconds % 60).toFixed(2);
        return minutes + '分' + secs + '秒';
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return hours + '小时' + minutes + '分钟';
    }
}

/**
 * 复制文本到剪贴板
 * @param {string} text - 要复制的文本
 */
function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            showSuccess('已复制到剪贴板');
        }).catch(() => {
            showError('复制失败');
        });
    } else {
        // 降级方案
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        showSuccess('已复制到剪贴板');
    }
}

/**
 * 获取 URL 查询参数
 * @param {string} param - 参数名
 * @returns {string|null} 参数值
 */
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

/**
 * 本地存储操作
 */
const Storage = {
    set: (key, value) => {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (e) {
            console.error('存储失败:', e);
        }
    },
    get: (key) => {
        try {
            const value = localStorage.getItem(key);
            return value ? JSON.parse(value) : null;
        } catch (e) {
            console.error('读取存储失败:', e);
            return null;
        }
    },
    remove: (key) => {
        try {
            localStorage.removeItem(key);
        } catch (e) {
            console.error('删除存储失败:', e);
        }
    },
    clear: () => {
        try {
            localStorage.clear();
        } catch (e) {
            console.error('清空存储失败:', e);
        }
    }
};
