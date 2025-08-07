import torch

def train(model, train_loader, val_loader, optimizer, criterion, num_epochs = 10, device = "cuda", patience = 5):
    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for i, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, inputs)

            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
    
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, captions in val_loader:
                images = images.to(device)
                captions = captions.to(device)

                inputs = captions[:, :-1]
                targets = captions[:, 1:]

                outputs = model(images, inputs)
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), targets.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            print(f"New best val_loss: {avg_val_loss}, Saving model...")
            best_val_loss = avg_val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improve_count += 1
            print(f"No improvement: {avg_val_loss} ({no_improve_count}/{patience})")

            if no_improve_count >= patience:
                print("Early stopping triggered.")
                break

    print(f"Epoch {epoch+1} completed. Avg Loss: {avg_train_loss:.4f}, Val Loss:   {best_val_loss:.4f}")