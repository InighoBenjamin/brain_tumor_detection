package com.braintumor.entity;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDate;
import java.time.LocalDateTime;

@Data
@Entity
@Table(name = "mri_meta_data")
public class MriMetaData {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer mriId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    private Patient patient;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "lab_id", nullable = false)
    private Lab lab;

    @Column(nullable = false, length = 255)
    private String mriPath;           // path to uploaded H5 file

    @Column(nullable = false)
    private LocalDate scanDate;

    private Integer volumeId;         // BraTS volume index (optional)
    private Integer sliceCount;       // total slices in the volume

    @Column(length = 10)
    private String modality = "multi"; // T1 / T2 / FLAIR / multi

    @Column(columnDefinition = "TEXT")
    private String notes;

    @Column(updatable = false)
    private LocalDateTime uploadedAt = LocalDateTime.now();
}
