function showError(msg) {
    const el = document.createElement("div");
    el.className = "message error";
    el.textContent = "Error: " + msg;
    document.getElementById("messages").appendChild(el);
}

var pendingImages = [];

// Max width/height for attached images; larger images are scaled down before upload.
var IMAGE_MAX_SIZE = 512;
var IMAGE_JPEG_QUALITY = 0.82;

/**
 * Resize an image (data URL or blob URL) so the longest side is at most maxSize.
 * Returns a Promise that resolves to a data URL (image/jpeg).
 */
function resizeImageToDataUrl(src, maxSize, quality) {
    return new Promise(function (resolve, reject) {
        var img = new Image();
        img.onload = function () {
            var w = img.naturalWidth;
            var h = img.naturalHeight;
            if (w <= maxSize && h <= maxSize) {
                var c = document.createElement("canvas");
                c.width = w;
                c.height = h;
                var ctx = c.getContext("2d");
                ctx.drawImage(img, 0, 0);
                resolve(c.toDataURL("image/jpeg", quality));
                return;
            }
            var scale = maxSize / Math.max(w, h);
            var nw = Math.round(w * scale);
            var nh = Math.round(h * scale);
            var canvas = document.createElement("canvas");
            canvas.width = nw;
            canvas.height = nh;
            var ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, w, h, 0, 0, nw, nh);
            resolve(canvas.toDataURL("image/jpeg", quality));
        };
        img.onerror = function () { reject(new Error("Failed to load image")); };
        img.src = src;
    });
}

function setBusy(busy) {
    var input = document.getElementById("input");
    var btn = document.getElementById("sendBtn");
    var attachBtn = document.getElementById("attachBtn");
    input.disabled = busy;
    btn.disabled = busy;
    attachBtn.disabled = busy;
}

function showThinking() {
    var div = document.createElement("div");
    div.id = "thinkingIndicator";
    div.className = "message thinking";
    div.textContent = "Thinking...";
    document.getElementById("messages").appendChild(div);
}

function hideThinking() {
    var el = document.getElementById("thinkingIndicator");
    if (el) el.remove();
}

function scrollMessagesToBottom() {
    var el = document.getElementById("messages");
    if (el) el.scrollTop = el.scrollHeight;
}

function dataUrlToBase64(dataUrl) {
    var idx = dataUrl.indexOf(",");
    return idx !== -1 ? dataUrl.slice(idx + 1) : dataUrl;
}

function renderAttachments() {
    var el = document.getElementById("attachments");
    el.innerHTML = "";
    pendingImages.forEach(function (dataUrl, i) {
        var wrap = document.createElement("div");
        wrap.className = "attachment-preview";
        var img = document.createElement("img");
        img.src = dataUrl;
        img.alt = "Attached";
        var remove = document.createElement("button");
        remove.type = "button";
        remove.className = "attachment-remove";
        remove.textContent = "\u00D7";
        remove.title = "Remove";
        remove.addEventListener("click", function () {
            pendingImages.splice(i, 1);
            renderAttachments();
        });
        wrap.appendChild(img);
        wrap.appendChild(remove);
        el.appendChild(wrap);
    });
}

async function send() {
    var input = document.getElementById("input");
    var text = input.value.trim();
    if (!text && pendingImages.length === 0) return;

    var imagesForRequest = pendingImages.map(dataUrlToBase64);
    var userMsgEl = addMessage("You", text || "(image)", "user", pendingImages.slice());
    input.value = "";
    pendingImages.length = 0;
    renderAttachments();

    setBusy(true);
    showThinking();
    scrollMessagesToBottom();

    try {
        const res = await fetch("/chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: text || "",
                images: imagesForRequest.length ? imagesForRequest : undefined
            })
        });

        if (!res.ok) {
            const errBody = await res.text();
            let errMsg = "Server returned " + res.status + ". " + (res.statusText || "");
            try {
                const d = JSON.parse(errBody);
                if (d.error) errMsg = d.error;
            } catch (_) {}
            showError(errMsg);
            return;
        }

        hideThinking();
        var container = document.getElementById("messages");
        var msgDiv = document.createElement("div");
        msgDiv.className = "message assistant";
        var label = document.createElement("strong");
        label.textContent = "Assistant: ";
        msgDiv.appendChild(label);
        var bodyDiv = document.createElement("div");
        bodyDiv.className = "message-body";
        msgDiv.appendChild(bodyDiv);
        container.appendChild(msgDiv);
        scrollMessagesToBottom();

        var thinkingSection = document.createElement("div");
        thinkingSection.className = "message-thinking-section";
        var thinkingHeader = document.createElement("button");
        thinkingHeader.type = "button";
        thinkingHeader.className = "message-thinking-toggle";
        thinkingHeader.setAttribute("aria-expanded", "true");
        thinkingHeader.innerHTML = "<span class=\"message-thinking-arrow\"></span> Thinking";
        var thinkingContent = document.createElement("div");
        thinkingContent.className = "message-thinking-content";
        var thinkingText = document.createElement("div");
        thinkingText.className = "message-thinking";
        thinkingContent.appendChild(thinkingText);
        thinkingSection.appendChild(thinkingHeader);
        thinkingSection.appendChild(thinkingContent);
        thinkingHeader.addEventListener("click", function () {
            var collapsed = thinkingSection.classList.toggle("collapsed");
            thinkingHeader.setAttribute("aria-expanded", collapsed ? "false" : "true");
        });
        var responseDiv = document.createElement("div");
        responseDiv.className = "message-response";
        responseDiv.style.display = "none";
        bodyDiv.appendChild(thinkingSection);
        bodyDiv.appendChild(responseDiv);

        var buffer = "";
        var fullText = "";
        var sources = [];
        var showingThinking = true;
        var hasAnyThinking = false;
        var reader = res.body.getReader();
        var decoder = new TextDecoder();

        function yieldToPaint() {
            return new Promise(function (r) { setTimeout(r, 0); });
        }

        while (true) {
            var chunk = await reader.read();
            if (chunk.done) break;
            buffer += decoder.decode(chunk.value, { stream: true });
            var parts = buffer.split("\n\n");
            buffer = parts.pop() || "";

            for (var i = 0; i < parts.length; i++) {
                var line = parts[i].trim();
                if (line.startsWith("data: ")) {
                    try {
                        var event = JSON.parse(line.slice(6));
                        if (event.searching) {
                            thinkingText.textContent += "Searching the web...\n\n";
                            thinkingContent.scrollTop = thinkingContent.scrollHeight;
                            hasAnyThinking = true;
                            scrollMessagesToBottom();
                        } else if (event.thinking !== undefined) {
                            if (showingThinking) {
                                hasAnyThinking = true;
                                thinkingText.textContent += event.thinking;
                                thinkingContent.scrollTop = thinkingContent.scrollHeight;
                                scrollMessagesToBottom();
                                await yieldToPaint();
                            }
                        } else if (event.token !== undefined) {
                            if (showingThinking) {
                                showingThinking = false;
                                thinkingSection.classList.add("collapsed");
                                thinkingHeader.setAttribute("aria-expanded", "false");
                                responseDiv.style.display = "";
                                fullText = event.token;
                                responseDiv.textContent = fullText;
                                scrollMessagesToBottom();
                            } else {
                                responseDiv.style.display = "";
                                fullText += event.token;
                                responseDiv.textContent = fullText;
                                scrollMessagesToBottom();
                            }
                        } else if (event.done) {
                            if (event.final !== undefined) {
                                fullText = event.final;
                                responseDiv.style.display = "";
                                responseDiv.textContent = fullText;
                            }
                            if (event.error) {
                                responseDiv.classList.add("error");
                            }
                            sources = event.sources || [];
                            if (event.user_timestamp && userMsgEl) addRemoveButton(userMsgEl, event.user_timestamp);
                            if (event.assistant_timestamp) addRemoveButton(msgDiv, event.assistant_timestamp);
                            if (!hasAnyThinking) {
                                thinkingSection.style.display = "none";
                            }
                            scrollMessagesToBottom();
                        }
                    } catch (e) {
                        console.warn("Parse SSE event failed", e);
                    }
                }
            }
        }

        if (typeof marked !== "undefined" && fullText) {
            responseDiv.innerHTML = marked.parse(fullText);
        }
        if (sources && sources.length) {
            addSources(sources);
        }
    } catch (err) {
        showError("Could not reach the server. Check that it is running and try again.");
        console.error(err);
    } finally {
        hideThinking();
        setBusy(false);
    }
}

function addRemoveButton(msgEl, timestamp) {
    if (!timestamp || msgEl.querySelector(".message-remove")) return;
    msgEl.setAttribute("data-memory-timestamp", timestamp);
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "message-remove";
    btn.textContent = "Remove from memory";
    btn.title = "Delete this message from conversation memory";
    btn.addEventListener("click", function () {
        removeFromMemory(timestamp, msgEl);
    });
    msgEl.appendChild(btn);
}

function addMessage(sender, text, cls, attachedImages, timestamp) {
    var div = document.createElement("div");
    div.className = "message " + cls;
    if (cls === "assistant" && typeof marked !== "undefined") {
        var label = document.createElement("strong");
        label.textContent = sender + ": ";
        div.appendChild(label);
        var body = document.createElement("div");
        body.className = "message-body";
        body.innerHTML = marked.parse(text);
        div.appendChild(body);
    } else {
        var head = document.createElement("span");
        head.textContent = sender + ": ";
        div.appendChild(head);
        if (text) {
            var textNode = document.createTextNode(text);
            div.appendChild(textNode);
        }
        if (attachedImages && attachedImages.length) {
            var imgWrap = document.createElement("div");
            imgWrap.className = "message-images";
            attachedImages.forEach(function (dataUrl) {
                var im = document.createElement("img");
                im.src = dataUrl;
                im.alt = "Attached";
                im.className = "message-thumb";
                imgWrap.appendChild(im);
            });
            div.appendChild(imgWrap);
        }
    }
    if (timestamp) addRemoveButton(div, timestamp);
    document.getElementById("messages").appendChild(div);
    return div;
}

async function removeFromMemory(timestamp, msgEl) {
    try {
        var res = await fetch("/memory/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ timestamp: timestamp })
        });
        if (res.ok && msgEl && msgEl.parentNode) {
            msgEl.remove();
        } else if (!res.ok) {
            var d = await res.json().catch(function () { return {}; });
            showError(d.error || "Failed to remove from memory");
        }
    } catch (e) {
        showError("Could not reach the server.");
    }
}

async function clearAllMemory() {
    if (!confirm("Clear all conversation memory? This cannot be undone.")) return;
    try {
        var res = await fetch("/memory/clear", { method: "POST" });
        if (res.ok) {
            document.getElementById("messages").innerHTML = "";
        } else {
            showError("Failed to clear memory");
        }
    } catch (e) {
        showError("Could not reach the server.");
    }
}

async function loadMemory() {
    try {
        var res = await fetch("/memory/recent");
        if (!res.ok) return;
        var data = await res.json();
        var entries = data.entries || [];
        entries.forEach(function (entry) {
            var sender = entry.role === "user" ? "You" : "Assistant";
            var cls = entry.role === "user" ? "user" : "assistant";
            var text = entry.content || "";
            if (entry.image_context) text += "\n[Image: " + entry.image_context + "]";
            addMessage(sender, text, cls, null, entry.timestamp);
        });
    } catch (e) {
        console.warn("Could not load memory", e);
    }
}

function addSources(urls) {
    const div = document.createElement("div");
    div.className = "sources";
    const markdown = "**Web sources:**\n\n" + urls.map(function (url) {
        return "- [" + url + "](" + url + ")";
    }).join("\n");
    if (typeof marked !== "undefined") {
        div.innerHTML = marked.parse(markdown);
    } else {
        div.textContent = "Web sources:\n" + urls.join("\n");
    }
    document.getElementById("messages").appendChild(div);
}

function addImageFromFile(file) {
    if (!file.type.startsWith("image/")) return;
    var reader = new FileReader();
    reader.onload = function () {
        var dataUrl = reader.result;
        resizeImageToDataUrl(dataUrl, IMAGE_MAX_SIZE, IMAGE_JPEG_QUALITY)
            .then(function (resized) {
                pendingImages.push(resized);
                renderAttachments();
            })
            .catch(function (err) {
                console.warn("Resize failed, using original", err);
                pendingImages.push(dataUrl);
                renderAttachments();
            });
    };
    reader.readAsDataURL(file);
}

document.addEventListener("DOMContentLoaded", function () {
    loadMemory().then(function () { scrollMessagesToBottom(); });

    document.getElementById("clearMemoryBtn").addEventListener("click", clearAllMemory);

    var fileInput = document.getElementById("fileInput");
    var attachBtn = document.getElementById("attachBtn");
    attachBtn.addEventListener("click", function () { fileInput.click(); });
    fileInput.addEventListener("change", function () {
        var files = fileInput.files;
        for (var i = 0; i < files.length; i++) addImageFromFile(files[i]);
        fileInput.value = "";
    });

    var input = document.getElementById("input");
    input.addEventListener("paste", function (e) {
        var items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (var i = 0; i < items.length; i++) {
            if (items[i].type.indexOf("image") !== -1) {
                e.preventDefault();
                addImageFromFile(items[i].getAsFile());
                return;
            }
        }
    });

    document.getElementById("sendBtn").addEventListener("click", send);
    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
            e.preventDefault();
            send();
        }
    });
});
