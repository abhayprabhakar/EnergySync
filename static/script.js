// Chart rendering
async function renderPowerConsumptionChart() {
    try {
        const response = await axios.get('/power_data');
        const powerData = response.data;

        // Soft, muted color palette
        const softColors = powerData.devices.map((_, index) => 
            `hsl(220, 50%, ${70 - (index * 5)}%)`
        );

        const ctx = document.getElementById('powerConsumptionChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: powerData.devices,
                datasets: [{
                    label: 'Power Consumption',
                    data: powerData.consumption,
                    backgroundColor: softColors,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Consumption Units',
                            color: '#6b7280'
                        },
                        ticks: {
                            color: '#6b7280'
                        }
                    },
                    x: {
                        ticks: {
                            color: '#6b7280'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });

        // Device Breakdown
        const breakdownHtml = powerData.devices.map((device, index) => `
            <div class="bg-white rounded-lg shadow-sm p-4 text-center transition hover:shadow-md">
                <div class="text-md font-medium text-gray-700">${device}</div>
                <div class="text-blue-600 font-semibold">${powerData.consumption[index].toFixed(2)} Units</div>
            </div>
        `).join('');
        
        document.getElementById('deviceBreakdown').innerHTML = breakdownHtml;
        document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
    } catch (error) {
        console.error('Error fetching power data:', error);
    }
}

// Chat functionality
async function sendMessage() {
    const messageInput = document.getElementById('messageInput');
    const chatMessages = document.getElementById('chatMessages');
    const message = messageInput.value.trim();

    if (!message) return;

    // User message
    const userMessageEl = document.createElement('div');
    userMessageEl.classList.add('bg-blue-50', 'rounded-xl', 'p-4', 'text-right');
    userMessageEl.innerHTML = `
        <p class="text-gray-800">
            <i class="fas fa-user mr-2 text-blue-400"></i>
            ${message}
        </p>
    `;
    chatMessages.appendChild(userMessageEl);

    try {
        const response = await axios.post('/chat', { message });
        
        // Bot message
        const botMessageEl = document.createElement('div');
        botMessageEl.classList.add('bg-white', 'rounded-xl', 'p-4', 'shadow-sm');
        botMessageEl.innerHTML = `
            <p class="text-gray-700">
                <i class="fas fa-robot mr-2 text-green-400"></i>
                ${response.data.response}
            </p>
        `;
        chatMessages.appendChild(botMessageEl);

        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        messageInput.value = '';
    } catch (error) {
        console.error('Error sending message:', error);
    }
}

// Event Listeners
document.getElementById('sendButton').addEventListener('click', sendMessage);
document.getElementById('messageInput').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

// Initial loads
renderPowerConsumptionChart();
setInterval(renderPowerConsumptionChart, 60000); // Refresh chart every minute