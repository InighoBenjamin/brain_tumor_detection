package com.braintumor.controller;

import com.braintumor.entity.MriMetaData;
import com.braintumor.service.MriService;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.time.LocalDate;
import java.util.List;

@RestController
@RequestMapping("/api/mri")
@RequiredArgsConstructor
public class MriController {

    private final MriService mriService;

    /**
     * POST /api/mri/upload
     * Upload an MRI H5 file. AI inference starts automatically.
     * Accessible by: doctor, lab_staff
     */
    @PostMapping("/upload")
    @PreAuthorize("hasAnyRole('doctor','lab_staff','admin')")
    public ResponseEntity<MriMetaData> upload(
            @RequestParam("file")       MultipartFile file,
            @RequestParam("patientId")  Integer patientId,
            @RequestParam("labId")      Integer labId,
            @RequestParam("scanDate")   @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate scanDate,
            @RequestParam(value = "notes", required = false) String notes
    ) throws IOException {
        MriMetaData mri = mriService.uploadAndProcess(patientId, labId, file, scanDate, notes);
        return ResponseEntity.ok(mri);
    }

    /**
     * GET /api/mri/patient/{patientId}
     * List all MRI scans for a patient.
     */
    @GetMapping("/patient/{patientId}")
    @PreAuthorize("hasAnyRole('doctor','radiologist','admin')")
    public ResponseEntity<List<MriMetaData>> getByPatient(@PathVariable Integer patientId) {
        return ResponseEntity.ok(mriService.getByPatient(patientId));
    }

    /**
     * GET /api/mri/{mriId}
     * Get a single MRI record.
     */
    @GetMapping("/{mriId}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<MriMetaData> getById(@PathVariable Integer mriId) {
        return ResponseEntity.ok(mriService.getById(mriId));
    }
}
