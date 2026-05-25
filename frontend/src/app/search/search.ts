import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-search',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './search.html',
  styleUrl: './search.css'
})
export class SearchComponent {
  query = '';
  answer = '';
  sources: any[] = [];
  uploadStatus = '';
  selectedFile: File | null = null;

  constructor(private http: HttpClient) {}

  onFileSelected(event: any) {
    this.selectedFile = event.target.files[0];
  }

  uploadFile() {
    if (!this.selectedFile) return;
    const formData = new FormData();
    formData.append('file', this.selectedFile);
    this.http.post('http://localhost:8000/ingest', formData)
      .subscribe(() => this.uploadStatus = 'Uploaded successfully');
  }

  search() {
  this.http.post<any>('http://localhost:8000/search', { query: this.query, top_k: 3, doc_ids: [] })
    .subscribe(res => {
      this.answer = res.answer;
      this.sources = res.sources || [];
    });
}
}
