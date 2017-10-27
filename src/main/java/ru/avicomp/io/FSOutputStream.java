package ru.avicomp.io;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Objects;

import com.google.common.io.LittleEndianDataOutputStream;


/**
 * To write byte data in c++(linux) little endian order.
 * Covers only primitives.
 *
 * @see com.google.common.io.LittleEndianDataOutputStream
 * Created by @szuev on 26.10.2017.
 */
public class FSOutputStream extends FilterOutputStream {

    public FSOutputStream(OutputStream out) {
        super(wrap(out));
    }

    private static LittleEndianDataOutputStream wrap(OutputStream out) {
        Objects.requireNonNull(out, "Null output steam specified.");
        return out instanceof LittleEndianDataOutputStream ? (LittleEndianDataOutputStream) out : new LittleEndianDataOutputStream(out);
    }

    private LittleEndianDataOutputStream out() {
        return (LittleEndianDataOutputStream) out;
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        out.write(b, off, len);
    }

    public void writeBoolean(boolean v) throws IOException {
        out().writeBoolean(v);
    }

    public void writeByte(int v) throws IOException {
        out().writeByte(v);
    }

    public void writeDouble(double v) throws IOException {
        out().writeDouble(v);
    }

    public void writeFloat(float v) throws IOException {
        out().writeFloat(v);
    }

    public void writeInt(int v) throws IOException {
        out().writeInt(v);
    }

    public void writeLong(long v) throws IOException {
        out().writeLong(v);
    }

    public void writeShort(int v) throws IOException {
        out().writeShort(v);
    }

    @Override
    public void close() throws IOException {
        out.close();
    }
}
