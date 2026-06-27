document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('add-car-form');
    const imageInput = document.getElementById('image');
    const imagePreview = document.getElementById('image-preview');
    const submitBtn = document.getElementById('submit-btn');
    const spinner = document.getElementById('loading-spinner');
    const btnText = submitBtn.querySelector('span');
    const messageDiv = document.getElementById('form-message');
    const inventoryList = document.getElementById('inventory-list');
    const leadsList = document.getElementById('leads-list');

    // Load inventory and leads on startup
    fetchInventory();
    fetchLeads();

    // Image Preview
    imageInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                imagePreview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            imagePreview.style.display = 'none';
            imagePreview.innerHTML = '';
        }
    });

    // Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading state
        submitBtn.disabled = true;
        btnText.textContent = 'Uploading...';
        spinner.classList.remove('hidden');
        messageDiv.classList.add('hidden');
        messageDiv.className = 'message hidden';

        const formData = new FormData(form);

        try {
            const response = await fetch('/api/admin/add_car', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                showMessage('Car successfully added to AI inventory!', 'success');
                form.reset();
                imagePreview.style.display = 'none';
                fetchInventory(); // Refresh the list
            } else {
                showMessage(`Error: ${result.detail || 'Failed to add car'}`, 'error');
            }
        } catch (error) {
            showMessage(`Network Error: ${error.message}`, 'error');
        } finally {
            // Restore UI state
            submitBtn.disabled = false;
            btnText.textContent = 'Add to Inventory';
            spinner.classList.add('hidden');
        }
    });

    function showMessage(text, type) {
        messageDiv.textContent = text;
        messageDiv.className = `message ${type}`;
    }

    // Fetch and display inventory
    async function fetchInventory() {
        try {
            const response = await fetch('/api/admin/inventory');
            if (response.ok) {
                const data = await response.json();
                renderInventory(data.inventory);
            } else {
                inventoryList.innerHTML = '<div class="loading-text">Failed to load inventory.</div>';
            }
        } catch (error) {
            inventoryList.innerHTML = '<div class="loading-text">Network error loading inventory.</div>';
        }
    }

    function renderInventory(items) {
        if (!items || items.length === 0) {
            inventoryList.innerHTML = '<div class="loading-text">No cars currently in inventory.</div>';
            return;
        }

        inventoryList.innerHTML = '';
        items.forEach(item => {
            const meta = item.payload.metadata;
            
            const div = document.createElement('div');
            div.className = 'inventory-item';
            div.innerHTML = `
                <div class="inventory-info">
                    <h3>${meta.year} ${meta.make} ${meta.model}</h3>
                    <p>Ksh ${meta.price_ksh.toLocaleString()} | ${meta.mileage_km.toLocaleString()} km</p>
                </div>
                <button class="delete-btn" data-id="${item.id}">Delete</button>
            `;
            
            // Add delete listener
            const deleteBtn = div.querySelector('.delete-btn');
            deleteBtn.addEventListener('click', () => deleteCar(item.id, div));
            
            inventoryList.appendChild(div);
        });
    }

    async function deleteCar(carId, elementToRemove) {
        if (!confirm('Are you sure you want to remove this car?')) return;

        const btn = elementToRemove.querySelector('.delete-btn');
        btn.textContent = '...';
        btn.disabled = true;

        try {
            const response = await fetch(`/api/admin/delete_car/${carId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                elementToRemove.remove();
                if (inventoryList.children.length === 0) {
                    inventoryList.innerHTML = '<div class="loading-text">No cars currently in inventory.</div>';
                }
            } else {
                alert('Failed to delete car.');
                btn.textContent = 'Delete';
                btn.disabled = false;
            }
        } catch (error) {
            alert('Network error while deleting.');
            btn.textContent = 'Delete';
            btn.disabled = false;
        }
    }

    // Fetch and display leads
    async function fetchLeads() {
        try {
            const response = await fetch('/api/admin/leads');
            if (response.ok) {
                const data = await response.json();
                renderLeads(data.leads);
            } else {
                leadsList.innerHTML = '<tr><td colspan="7" class="loading-text text-center">Failed to load leads.</td></tr>';
            }
        } catch (error) {
            leadsList.innerHTML = '<tr><td colspan="7" class="loading-text text-center">Network error loading leads.</td></tr>';
        }
    }

    function renderLeads(leads) {
        if (!leads || leads.length === 0) {
            leadsList.innerHTML = '<tr><td colspan="7" class="loading-text text-center">No test drives scheduled yet.</td></tr>';
            return;
        }

        leadsList.innerHTML = '';
        leads.forEach(lead => {
            const date = new Date(lead.created_at).toLocaleString();
            
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${date}</td>
                <td><strong>${lead.customer_name}</strong></td>
                <td>${lead.customer_contact}</td>
                <td>${lead.car_of_interest}</td>
                <td>${lead.preferred_date_time}</td>
                <td><span class="status-badge status-new">${lead.status}</span></td>
                <td>
                    <a href="/api/admin/leads/${lead.id}/ics" class="calendar-btn" download>📅 Add to Calendar</a>
                </td>
            `;
            leadsList.appendChild(tr);
        });
    }
});
