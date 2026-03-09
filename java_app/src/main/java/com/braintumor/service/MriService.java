package com.braintumor.service;

import com.braintumor.entity.*;
import com.braintumor.repository.MriRepository;
import com.braintumor.repository.PatientRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.*;
import java.time.LocalDate;
import java.util.List;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class MriService {

    private final MriRepository mriRepository;
    private final PatientRepository patientRepository;
    private final AiInferenceService aiInferenceService;

    @Value("${app.upload.dir}")
    private String uploadDir;

    /**
     * Save an uploaded MRI H5 file to disk, register it in DB,
     * and trigger async AI inference.
     */
    public MriMetaData uploadAndProcess(
            Integer patientId,
            Integer labId,
            MultipartFile file,
            LocalDate scanDate,
            String notes
    ) throws IOException {

        Patient patient = patientRepository.findById(patientId)
            .orElseThrow(() -> new IllegalArgumentException("Patient not found: " + patientId));

        // Save file to disk with a unique filename
        String filename = UUID.randomUUID() + "_" + file.getOriginalFilename();
        Path dest = Paths.get(uploadDir, filename);
        Files.createDirectories(dest.getParent());
        Files.copy(file.getInputStream(), dest, StandardCopyOption.REPLACE_EXISTING);

        // Build Lab reference (minimal — just ID)
        Lab lab = new Lab();
        lab.setLabId(labId);

        // Persist MRI record
        MriMetaData mri = new MriMetaData();
        mri.setPatient(patient);
        mri.setLab(lab);
        mri.setMriPath(dest.toAbsolutePath().toString());
        mri.setScanDate(scanDate);
        mri.setNotes(notes);
        mri = mriRepository.save(mri);

        // Trigger AI inference asynchronously
        aiInferenceService.runInference(mri);

        log.info("MRI id={} uploaded for patient={}, inference started", mri.getMriId(), patientId);
        return mri;
    }

    public List<MriMetaData> getByPatient(Integer patientId) {
        return mriRepository.findByPatient_PatientId(patientId);
    }

    public MriMetaData getById(Integer mriId) {
        return mriRepository.findById(mriId)
            .orElseThrow(() -> new IllegalArgumentException("MRI not found: " + mriId));
    }
}
