(() => {
    const activityCanvas = document.getElementById("activityChart");
    if (!activityCanvas || typeof Chart === "undefined") {
        return;
    }

    const activityLabels = JSON.parse(activityCanvas.dataset.labels || "[]");
    const activityValues = JSON.parse(activityCanvas.dataset.values || "[]");
    new Chart(activityCanvas, {
        type: "line",
        data: {
            labels: activityLabels,
            datasets: [{
                label: "Articles analyzed",
                data: activityValues,
                borderColor: "#5267e8",
                backgroundColor: "rgba(82, 103, 232, 0.10)",
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false } },
                y: {
                    beginAtZero: true,
                    suggestedMax: Math.max(1, ...activityValues),
                    ticks: { precision: 0 }
                }
            }
        }
    });

    const distributionCanvas = document.getElementById("distributionChart");
    if (!distributionCanvas) {
        return;
    }
    new Chart(distributionCanvas, {
        type: "doughnut",
        data: {
            labels: ["Real news", "Fake news"],
            datasets: [{
                data: [Number(distributionCanvas.dataset.real), Number(distributionCanvas.dataset.fake)],
                backgroundColor: ["#16a36a", "#ef5350"],
                borderWidth: 0,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "72%",
            plugins: { legend: { display: false } }
        }
    });
})();
