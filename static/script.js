document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const sendBtn = document.getElementById('send-btn');
    
    // Maintain chat history state
    let chatHistory = [];

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = userInput.value.trim();
        if (!message) return;

        // Add user message to UI and history
        appendMessage('user', message);
        chatHistory.push({ role: 'user', content: message });
        
        userInput.value = '';
        sendBtn.disabled = true;

        // Add typing indicator
        const typingId = appendTypingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                // Send current message and history (excluding the current message just pushed)
                body: JSON.stringify({ 
                    message: message,
                    history: chatHistory.slice(0, -1).slice(-6) // Keep last 6 messages for context limit
                })
            });

            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            
            // Remove typing indicator
            removeElement(typingId);

            // Add bot response to UI and history
            appendMessage('bot', data.response, data.image_url);
            chatHistory.push({ role: 'bot', content: data.response });

        } catch (error) {
            console.error('Error:', error);
            removeElement(typingId);
            const errorMsg = 'Sorry, I am having trouble connecting to the server right now. Please try again.';
            appendMessage('bot', errorMsg);
            chatHistory.push({ role: 'bot', content: errorMsg });
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
        }
    });

    function appendMessage(sender, text, imageUrl = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Parse markdown if bot (sanitized using DOMPurify to protect against XSS)
        if (sender === 'bot') {
            contentDiv.innerHTML = DOMPurify.sanitize(marked.parse(text));
        } else {
            contentDiv.textContent = text;
        }

        if (imageUrl && imageUrl.startsWith('http')) {
            const img = document.createElement('img');
            img.src = imageUrl;
            img.className = 'car-image';
            img.alt = 'Car Image';
            contentDiv.appendChild(img);
        }

        messageDiv.appendChild(contentDiv);
        chatBox.appendChild(messageDiv);
        scrollToBottom();
    }

    function appendTypingIndicator() {
        const id = 'typing-' + Date.now();
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.id = id;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            contentDiv.appendChild(dot);
        }

        messageDiv.appendChild(contentDiv);
        chatBox.appendChild(messageDiv);
        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
});
