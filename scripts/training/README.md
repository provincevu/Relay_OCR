# Training

Nhóm này định nghĩa, huấn luyện và đánh giá Layout model cho page 3+.

| Tệp | Trách nhiệm |
|---|---|
| `page3_layout_model.py` | Định nghĩa model, encode và predict dùng chung. |
| `train_page3_text_layout.py` | Huấn luyện LayoutLMv3 từ annotation hoặc embedding cache. |
| `prepare_page3_embeddings.py` | Tạo cache embedding trước khi huấn luyện. |
| `evaluate_page3_text_layout.py` | Đánh giá role và quan hệ KEY → VALUE. |

Nhóm này nhận annotation hoặc embedding cache và tạo checkpoint, metric hoặc kết quả đánh giá.
